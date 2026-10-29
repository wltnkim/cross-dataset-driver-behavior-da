"""Cross-dataset driver behavior recognition pipeline.

Provides unified data loading for OBC, Drive&Act, and 3MDAD datasets
with consistent class mapping for domain adaptation experiments.
"""

from cross_dataset.class_mapping_utils import load_class_mapping, get_unified_label_maps
