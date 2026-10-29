"""Utilities for cross-dataset class mapping.

Loads class_mapping.json and builds lookup tables for converting
dataset-specific labels to unified labels.
"""

import json
import os


def load_class_mapping(mapping_path=None):
    """Load the class mapping JSON."""
    if mapping_path is None:
        mapping_path = os.path.join(os.path.dirname(__file__), "class_mapping.json")
    with open(mapping_path) as f:
        return json.load(f)


def get_unified_label_maps(mapping_path=None):
    """Build per-dataset label mapping dicts.

    Returns:
        dict with keys 'obc', 'driveandact', 'threeMDAD'.
        Each value is a dict mapping dataset-specific class name -> unified_label (int).
        Classes not in the mapping are excluded (return None).
    """
    mapping = load_class_mapping(mapping_path)
    unified_classes = mapping["unified_classes"]

    obc_map = {}
    dna_map = {}
    mdad_map = {}
    dmd_map = {}

    for entry in unified_classes:
        ulabel = entry["unified_label"]

        for cls in entry["obc"]:
            obc_map[cls] = ulabel

        for cls in entry["driveandact"]:
            dna_map[cls] = ulabel

        if entry["threeMDAD"] is not None:
            for cls in entry["threeMDAD"]:
                mdad_map[cls] = ulabel

        if entry.get("dmd") is not None:
            for cls in entry["dmd"]:
                dmd_map[cls] = ulabel

    return {
        "obc": obc_map,
        "driveandact": dna_map,
        "threeMDAD": mdad_map,
        "dmd": dmd_map,
    }


def get_unified_class_names(mapping_path=None):
    """Return list of unified class names ordered by unified_label."""
    mapping = load_class_mapping(mapping_path)
    classes = sorted(mapping["unified_classes"], key=lambda x: x["unified_label"])
    return [c["name"] for c in classes]


def get_num_unified_classes(mapping_path=None):
    """Return number of unified classes."""
    mapping = load_class_mapping(mapping_path)
    return len(mapping["unified_classes"])
