"""Shared coverage/containment statistics + their closed-form unit tests.

Definitions match access_rebuttal_prep/tools/coverage_stats.py exactly so the new numbers are
comparable with last round's table:
  erank = exp(-sum p_i log p_i), p_i = lam_i / sum(lam), lam = eigenvalues of the CENTRED
          sample covariance (Roy & Vetterli)
  PR    = (sum lam)^2 / sum(lam^2)
  mpcd  = mean over i<j of 1 - cos(x_i, x_j), UNCENTRED
  nn_cos_dist(Q, R) = for each q in Q, min over r in R of (1 - cos(q, r))
"""
import numpy as np


def eig(X):
    Xc = X - X.mean(0, keepdims=True)
    C = (Xc.T @ Xc) / max(len(X) - 1, 1)
    return np.clip(np.linalg.eigvalsh(C), 0, None)


def erank(X):
    w = eig(X); s = w.sum()
    if s <= 0: return float("nan")
    p = w / s; p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def pratio(X):
    w = eig(X); d = (w ** 2).sum()
    return float((w.sum() ** 2) / d) if d > 0 else float("nan")


def _unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def mpcd(X):
    Xn = _unit(X); G = Xn @ Xn.T
    iu = np.triu_indices(len(X), 1)
    return float((1.0 - G[iu]).mean())


def cross_mean_cos_dist(A, B):
    """Mean 1-cos over ALL cross pairs (a in A, b in B)."""
    return float((1.0 - _unit(A) @ _unit(B).T).mean())


def nn_cos_dist(Q, R, exclude_self=False, block=2048):
    """For each row of Q, the cosine distance to its nearest neighbour in R.

    exclude_self=True treats Q and R as the SAME set and ignores the zero self-match
    (leave-one-out). Blocked so a large R does not build an n x m matrix at once.
    """
    Qn, Rn = _unit(Q), _unit(R)
    out = np.empty(len(Qn), dtype=np.float64)
    for s in range(0, len(Qn), block):
        e = min(s + block, len(Qn))
        D = 1.0 - Qn[s:e] @ Rn.T
        if exclude_self:
            for i in range(s, e):
                D[i - s, i] = np.inf
        out[s:e] = D.min(axis=1)
    return out


def selftest(verbose=True):
    """Closed-form checks. Every assertion has a known exact answer."""
    r = []
    rng = np.random.RandomState(0)

    # erank / PR recover the true rank of an isotropic subspace
    for k in (2, 5, 10):
        X = np.zeros((4000, 20)); X[:, :k] = rng.randn(4000, k)
        r.append(("erank rank-%d" % k, erank(X), k, 0.05))
        r.append(("PR rank-%d" % k, pratio(X), k, 0.05))

    # mpcd: orthonormal rows -> exactly 1.0 ; identical rows -> exactly 0.0
    r.append(("mpcd orthonormal", mpcd(np.eye(6)), 1.0, 1e-12))
    r.append(("mpcd identical", mpcd(np.tile(rng.randn(1, 20), (50, 1))), 0.0, 1e-9))

    # known angle: unit vectors 60 deg apart -> cos 0.5 -> distance 0.5
    a = np.array([[1.0, 0.0]]); b = np.array([[0.5, np.sqrt(3) / 2]])
    r.append(("mpcd 60deg", mpcd(np.vstack([a, b])), 0.5, 1e-12))
    r.append(("cross 60deg", cross_mean_cos_dist(a, b), 0.5, 1e-12))

    # nn distance: identical sets, self excluded -> nearest is the orthogonal partner (1.0)
    r.append(("nn orthonormal LOO", float(nn_cos_dist(np.eye(4), np.eye(4), True).mean()), 1.0, 1e-12))
    # nn distance: query equal to a reference row -> exactly 0
    R = rng.randn(50, 8); Q = R[[3, 17]]
    r.append(("nn exact match", float(nn_cos_dist(Q, R).max()), 0.0, 1e-12))
    # nn distance: scale invariance (cosine ignores magnitude)
    r.append(("nn scale-invariant", float(np.abs(nn_cos_dist(Q * 7.3, R) - nn_cos_dist(Q, R)).max()),
              0.0, 1e-9))
    # nn distance is bounded in [0, 2]
    Z = rng.randn(200, 16)
    d = nn_cos_dist(Z, -Z)
    r.append(("nn upper bound <=2", float(d.max() <= 2.0 + 1e-9), 1.0, 1e-12))

    bad = []
    for name, got, want, tol in r:
        ok = abs(got - want) <= tol
        if verbose:
            print("  %-22s got=%.6f want=%.6f %s" % (name, got, want, "OK" if ok else "**FAIL**"))
        if not ok: bad.append(name)
    return bad


if __name__ == "__main__":
    import sys
    print("closed-form self-test of stat_lib:")
    bad = selftest()
    print("SELFTEST", "PASS" if not bad else "FAIL %s" % bad)
    sys.exit(3 if bad else 0)
