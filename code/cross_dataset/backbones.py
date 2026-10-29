"""[ARR] Backbone ladder for the E3 group. NEW FILE - no frozen counterpart.

Three backbones behind one interface:
  videomae      the existing path, untouched (train_cross.VideoMAEDANN handles it)
  r2plus1d_18   torchvision R(2+1)D-18, Kinetics-400 weights, 112px input
  resnet50_tsn  torchvision ResNet-50, ImageNet weights, per-frame + temporal average pool

FRAME SAMPLING: none of this code samples frames. It consumes whatever the dataset produced.
The 51 -> 16 rule lives in the frozen `obc_dataset.py:138-142`
    sample_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
    # + optional temporal jitter when augment=True
which for 51 frames gives [0,3,6,10,13,16,20,23,26,30,33,36,40,43,46,50] - 16 distinct indices.
The memmap256 targets are ALREADY stored at 16 frames, so no 51->16 step applies to them.
That rule is REUSED as-is for every backbone; nothing here re-samples.

NORMALIZATION (REVIEWED DECISION, not an oversight): each backbone keeps the statistics its own
pretraining used -- ImageNet for VideoMAE and ResNet-50, the torchvision Kinetics-400 video
statistics for R(2+1)D-18. The original brief said "ImageNet mean/std for the CNNs"; the deviation
for R(2+1)D-18 was raised, reviewed and APPROVED by the user on 2026-08-29, on the grounds that
feeding ImageNet statistics to Kinetics-pretrained weights would mismatch their own pretraining.

The frozen datasets hand us tensors already normalised with ImageNet statistics
(obc_dataset.py:33-34, memmap_dataset.py:27-28). A backbone whose pretraining used different
statistics is therefore RE-normalised here - de-normalise with the dataset's constants, then
re-normalise with the backbone's own. That is exact, and `selftest()` proves it against the
closed form.
"""
import hashlib, os
import torch
import torch.nn as nn
import torch.nn.functional as F

# what the frozen datasets applied (obc_dataset.py:33-34, memmap_dataset.py:27-28)
DATASET_MEAN = (0.485, 0.456, 0.406)
DATASET_STD = (0.229, 0.224, 0.225)

# what each backbone's pretraining used
NORM = {
    "videomae":     {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
                     "size": 224, "why": "unchanged - the existing VideoMAE path"},
    "resnet50_tsn": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
                     "size": 224, "why": "ImageNet, per torchvision ResNet50_Weights"},
    "r2plus1d_18":  {"mean": (0.43216, 0.394666, 0.37645), "std": (0.22803, 0.22145, 0.216989),
                     "size": 112, "why": "torchvision R2Plus1D_18_Weights.KINETICS400_V1 "
                                         "canonical video statistics, NOT ImageNet"},
}
BACKBONES = tuple(NORM)


def _t(v, x):
    return torch.tensor(v, dtype=x.dtype, device=x.device).view(1, 1, 3, 1, 1)


def renormalize(x, backbone):
    """[B,T,C,H,W] normalised with DATASET_* -> normalised with NORM[backbone].

    x_raw = x * ds_std + ds_mean ; out = (x_raw - mean) / std.  No-op when they match.
    """
    n = NORM[backbone]
    if n["mean"] == DATASET_MEAN and n["std"] == DATASET_STD:
        return x
    raw = x * _t(DATASET_STD, x) + _t(DATASET_MEAN, x)
    return (raw - _t(n["mean"], x)) / _t(n["std"], x)


def resize_for(x, backbone):
    """Spatial resize to the backbone's expected side, inside its own transform."""
    s = NORM[backbone]["size"]
    if x.shape[-1] == s and x.shape[-2] == s:
        return x
    B, T, C, H, W = x.shape
    return F.interpolate(x.reshape(B * T, C, H, W), size=(s, s), mode="bilinear",
                         align_corners=False).reshape(B, T, C, s, s)


def preprocess(x, backbone):
    """The full per-backbone transform: renormalise, then resize."""
    return resize_for(renormalize(x, backbone), backbone)


class R2Plus1DFeatures(nn.Module):
    """torchvision r2plus1d_18 with the classifier removed. Input [B,T,C,H,W] -> [B,512]."""
    feat_dim = 512

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        w = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        m = r2plus1d_18(weights=w)
        m.fc = nn.Identity()
        self.net = m

    def forward(self, x):                       # [B,T,C,H,W]
        return self.net(x.permute(0, 2, 1, 3, 4))   # -> [B,C,T,H,W] -> [B,512]


class ResNet50TSNFeatures(nn.Module):
    """Per-frame ResNet-50 then temporal average pooling (TSN consensus).
    Input [B,T,C,H,W] -> [B,2048]."""
    feat_dim = 2048

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        w = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = resnet50(weights=w)
        m.fc = nn.Identity()
        self.net = m

    def forward(self, x):
        B, T, C, H, W = x.shape
        f = self.net(x.reshape(B * T, C, H, W))     # [B*T, 2048]
        return f.reshape(B, T, -1).mean(1)          # temporal average pooling


def build_feature_extractor(backbone, pretrained=True):
    if backbone == "r2plus1d_18":
        return R2Plus1DFeatures(pretrained)
    if backbone == "resnet50_tsn":
        return ResNet50TSNFeatures(pretrained)
    raise ValueError("build_feature_extractor does not handle %r (videomae uses the "
                     "existing train_cross path)" % backbone)


class CNNDANN(nn.Module):
    """DANN wiring for a CNN backbone, mirroring VideoMAEDANN's (logits, domain, feat) contract.

    grl_factory is injected by the caller (train_cross.GradientReversalLayer) so this module
    never imports train_cross - that would be circular.
    """

    def __init__(self, backbone, num_classes, grl_factory, pretrained=True, hidden=1024):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.features = build_feature_extractor(backbone, pretrained)
        d = self.features.feat_dim
        self.classifier = nn.Linear(d, num_classes)
        self.grl = grl_factory()
        self.domain_classifier = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden, 2))

    def forward(self, x, lambda_dann=0.0):
        x = preprocess(x, self.backbone)
        f = self.features(x)
        return self.classifier(f), self.domain_classifier(self.grl(f, lambda_dann)), f



class _Out:
    """Mimics the HF ModelOutput contract the non-DA path uses (`model(pixel_values=..).logits`)."""
    def __init__(self, logits): self.logits = logits


class CNNPlain(nn.Module):
    """Non-DA path (target_only / finetune) for a CNN backbone."""

    def __init__(self, backbone, num_classes, pretrained=True):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.features = build_feature_extractor(backbone, pretrained)
        self.classifier = nn.Linear(self.features.feat_dim, num_classes)

    def forward(self, pixel_values=None, x=None):
        t = pixel_values if pixel_values is not None else x
        return _Out(self.classifier(self.features(preprocess(t, self.backbone))))


# --------------------------------------------------------------------------------------
def selftest(verbose=True):
    """Closed-form checks on the per-backbone transform. Both sides, red included."""
    bad = []
    torch.manual_seed(0)
    raw = torch.rand(2, 16, 3, 224, 224)                    # a [0,1] clip
    dsn = (raw - _t(DATASET_MEAN, raw)) / _t(DATASET_STD, raw)   # what the dataset hands us

    for b in BACKBONES:
        n = NORM[b]
        want = (raw - _t(n["mean"], raw)) / _t(n["std"], raw)     # normalise the RAW clip directly
        got = renormalize(dsn, b)
        err = float((got - want).abs().max())
        ok = err < 1e-5
        if verbose: print("  renormalize %-13s max|err| = %.2e  %s" % (b, err, "OK" if ok else "**FAIL**"))
        if not ok: bad.append("renormalize:" + b)

    # RED SIDE 1: applying the WRONG backbone's stats must NOT match
    wrong = float((renormalize(dsn, "r2plus1d_18")
                   - (raw - _t(NORM["resnet50_tsn"]["mean"], raw)) / _t(NORM["resnet50_tsn"]["std"], raw)
                   ).abs().max())
    ok = wrong > 1e-2
    if verbose: print("  red: r2plus1d stats vs resnet stats differ by %.4f  %s" % (wrong, "OK" if ok else "**FAIL**"))
    if not ok: bad.append("red:stats-identical")

    # RED SIDE 2: videomae must be an exact no-op (its stats ARE the dataset's)
    noop = float((renormalize(dsn, "videomae") - dsn).abs().max())
    ok = noop == 0.0
    if verbose: print("  videomae renormalize is an exact no-op: %.1e  %s" % (noop, "OK" if ok else "**FAIL**"))
    if not ok: bad.append("videomae:not-noop")

    # sizes
    for b, want in (("videomae", 224), ("resnet50_tsn", 224), ("r2plus1d_18", 112)):
        s = preprocess(dsn, b).shape
        ok = s[-1] == want and s[-2] == want and s[:3] == dsn.shape[:3]
        if verbose: print("  preprocess %-13s -> %s (want %dpx)  %s" % (b, tuple(s), want, "OK" if ok else "**FAIL**"))
        if not ok: bad.append("size:" + b)

    # RED SIDE 3: resize must actually change r2plus1d and not the others
    if preprocess(dsn, "r2plus1d_18").shape[-1] == dsn.shape[-1]: bad.append("red:resize-noop")
    return bad


if __name__ == "__main__":
    import sys
    print("backbones.py closed-form self-test:")
    b = selftest()
    print("SELFTEST", "PASS" if not b else "FAIL %s" % b)
    sys.exit(3 if b else 0)
