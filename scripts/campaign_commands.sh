#!/usr/bin/env bash
# Training commands for the 78 revision-campaign runs (E1: source composition and mechanism pair,
# E2: zero-label reverse direction), extracted verbatim from the original launch scripts.
# Only the cluster wrapper, scheduler settings and logging around each command were removed.
# Exact arguments of every released run (including the benchmark runs of Tables 4-9, E3, E4
# and IR) are recorded in results/<run>/config.yaml.
#
# Usage: set DATA_ROOT to the folder that holds your copies of the datasets and the
# preprocessed memmaps (see README), then run individual lines from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA_ROOT:?set DATA_ROOT to your dataset root}"
export PYTHONPATH=$PWD/code TORCH_HOME=$PWD/weights

# E1_cdan_broad573__3mdad_f0
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset 3mdad \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_broad573__3mdad_f0 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_broad573__3mdad_f1
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset 3mdad \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_broad573__3mdad_f1 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_broad573__3mdad_f2
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset 3mdad \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_broad573__3mdad_f2 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_broad573__dmd_f0
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset dmd \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_broad573__dmd_f0 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_broad573__dmd_f1
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset dmd \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_broad573__dmd_f1 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_broad573__dmd_f2
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset dmd \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_broad573__dmd_f2 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_da573_random__3mdad_f0
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset 3mdad \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_da573_random__3mdad_f0 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_da573_random__3mdad_f1
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset 3mdad \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_da573_random__3mdad_f1 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_da573_random__3mdad_f2
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset 3mdad \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_da573_random__3mdad_f2 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_da573_random__dmd_f0
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset dmd \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_da573_random__dmd_f0 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_da573_random__dmd_f1
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset dmd \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_da573_random__dmd_f1 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_da573_random__dmd_f2
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset dmd \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_da573_random__dmd_f2 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_narrow573__3mdad_f0
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset 3mdad \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_narrow573__3mdad_f0 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_narrow573__3mdad_f1
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset 3mdad \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_narrow573__3mdad_f1 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_narrow573__3mdad_f2
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset 3mdad \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_narrow573__3mdad_f2 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_narrow573__dmd_f0
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset dmd \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_narrow573__dmd_f0 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_narrow573__dmd_f1
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset dmd \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_narrow573__dmd_f1 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_narrow573__dmd_f2
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset dmd \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_narrow573__dmd_f2 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_cdan_obc__3mdad_f0
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset 3mdad \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_obc__3mdad_f0 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_cdan_obc__3mdad_f1
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset 3mdad \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_obc__3mdad_f1 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_cdan_obc__3mdad_f2
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset 3mdad \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_obc__3mdad_f2 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_cdan_obc__dmd_f0
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset dmd \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_obc__dmd_f0 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_cdan_obc__dmd_f1
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset dmd \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_obc__dmd_f1 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_cdan_obc__dmd_f2
python -m cross_dataset.train_cross \
    --mode cdan \
    --target_dataset dmd \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_cdan_obc__dmd_f2 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_dann_broad573__3mdad_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_broad573__3mdad_f0 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_broad573__3mdad_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_broad573__3mdad_f1 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_broad573__3mdad_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_broad573__3mdad_f2 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_broad573__dmd_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_broad573__dmd_f0 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_broad573__dmd_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_broad573__dmd_f1 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_broad573__dmd_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_broad573__dmd_f2 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_da573_random__3mdad_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_da573_random__3mdad_f0 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_da573_random__3mdad_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_da573_random__3mdad_f1 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_da573_random__3mdad_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_da573_random__3mdad_f2 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_da573_random__dmd_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_da573_random__dmd_f0 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_da573_random__dmd_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_da573_random__dmd_f1 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_da573_random__dmd_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_da573_random__dmd_f2 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_narrow573__3mdad_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_narrow573__3mdad_f0 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_narrow573__3mdad_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_narrow573__3mdad_f1 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_narrow573__3mdad_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_narrow573__3mdad_f2 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_narrow573__dmd_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_narrow573__dmd_f0 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_narrow573__dmd_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_narrow573__dmd_f1 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_narrow573__dmd_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_narrow573__dmd_f2 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_dann_obc__3mdad_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_obc__3mdad_f0 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_dann_obc__3mdad_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_obc__3mdad_f1 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_dann_obc__3mdad_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_obc__3mdad_f2 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_dann_obc__dmd_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_obc__dmd_f0 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_dann_obc__dmd_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_obc__dmd_f1 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_dann_obc__dmd_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_dann_obc__dmd_f2 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_source_only_broad573__3mdad_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_broad573__3mdad_f0 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_broad573__3mdad_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_broad573__3mdad_f1 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_broad573__3mdad_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_broad573__3mdad_f2 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_broad573__dmd_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_broad573__dmd_f0 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_broad573__dmd_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_broad573__dmd_f1 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_broad573__dmd_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_broad573__dmd_f2 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/broad573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_da573_random__3mdad_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_da573_random__3mdad_f0 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_da573_random__3mdad_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_da573_random__3mdad_f1 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_da573_random__3mdad_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_da573_random__3mdad_f2 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_da573_random__dmd_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_da573_random__dmd_f0 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_da573_random__dmd_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_da573_random__dmd_f1 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_da573_random__dmd_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_da573_random__dmd_f2 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/da573_random.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_narrow573__3mdad_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_narrow573__3mdad_f0 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_narrow573__3mdad_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_narrow573__3mdad_f1 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_narrow573__3mdad_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_narrow573__3mdad_f2 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --source_manifest $PWD/manifests/narrow573.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json

# E1_source_only_obc__3mdad_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_obc__3mdad_f0 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_source_only_obc__3mdad_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_obc__3mdad_f1 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_source_only_obc__3mdad_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset 3mdad \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_obc__3mdad_f2 \
    --memmap_path $DATA_ROOT/memmap256/3mdad_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_source_only_obc__dmd_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_obc__dmd_f0 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_source_only_obc__dmd_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_obc__dmd_f1 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E1_source_only_obc__dmd_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset dmd \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.0 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E1_source_only_obc__dmd_f2 \
    --memmap_path $DATA_ROOT/memmap256/dmd_frames.npy \
    --memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --memmap_crop 224 \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E2_dann_3mdad_full153__obc_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset obc \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E2_dann_3mdad_full153__obc_f0 \
    --seed 42 \
    --source_manifest $PWD/manifests/3mdad_full153.csv \
    --source_memmap $DATA_ROOT/memmap256/3mdad_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E2_dann_3mdad_full153__obc_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset obc \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E2_dann_3mdad_full153__obc_f1 \
    --seed 43 \
    --source_manifest $PWD/manifests/3mdad_full153.csv \
    --source_memmap $DATA_ROOT/memmap256/3mdad_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E2_dann_3mdad_full153__obc_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset obc \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E2_dann_3mdad_full153__obc_f2 \
    --seed 44 \
    --source_manifest $PWD/manifests/3mdad_full153.csv \
    --source_memmap $DATA_ROOT/memmap256/3mdad_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/3mdad_metadata.json \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E2_dann_dmd_full2659__obc_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset obc \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E2_dann_dmd_full2659__obc_f0 \
    --seed 42 \
    --source_manifest $PWD/manifests/dmd_full2659.csv \
    --source_memmap $DATA_ROOT/memmap256/dmd_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E2_dann_dmd_full2659__obc_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset obc \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E2_dann_dmd_full2659__obc_f1 \
    --seed 43 \
    --source_manifest $PWD/manifests/dmd_full2659.csv \
    --source_memmap $DATA_ROOT/memmap256/dmd_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E2_dann_dmd_full2659__obc_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset obc \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E2_dann_dmd_full2659__obc_f2 \
    --seed 44 \
    --source_manifest $PWD/manifests/dmd_full2659.csv \
    --source_memmap $DATA_ROOT/memmap256/dmd_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/dmd_metadata.json \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E2_dann_driveandact_full2368__obc_f0
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset obc \
    --fold 0 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E2_dann_driveandact_full2368__obc_f0 \
    --seed 42 \
    --source_manifest $PWD/manifests/driveandact_full2368.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E2_dann_driveandact_full2368__obc_f1
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset obc \
    --fold 1 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E2_dann_driveandact_full2368__obc_f1 \
    --seed 43 \
    --source_manifest $PWD/manifests/driveandact_full2368.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# E2_dann_driveandact_full2368__obc_f2
python -m cross_dataset.train_cross \
    --mode dann \
    --target_dataset obc \
    --fold 2 \
    --epochs 50 \
    --batch_size 8 \
    --alpha_dann 0.5 \
    --target_label_ratio 0.0 \
    --patience 999 \
    --num_workers 4 \
    --instrument \
    --run_tag E2_dann_driveandact_full2368__obc_f2 \
    --seed 44 \
    --source_manifest $PWD/manifests/driveandact_full2368.csv \
    --source_memmap $DATA_ROOT/memmap256/driveandact_frames.npy \
    --source_memmap_metadata $DATA_ROOT/memmap256/driveandact_metadata.json \
    --obc_metadata $DATA_ROOT/UMTRI/Occupant_Behavior/code/features/keyframe_video_metadata_dualcam.pkl

# Selection-rule re-evaluation (four checkpoint-selection rules per run; needs the trained
# checkpoints of the runs above). Output feeds analysis/reeval/ and tools/reeval_table.py.
# --wall-seconds is the script's time budget; the original jobs passed their remaining scheduler time.
python tools/reeval_rules.py --target 3mdad --fold 0 --out analysis/reeval/3mdad_f0.json --wall-seconds 86400
python tools/reeval_rules.py --target 3mdad --fold 1 --out analysis/reeval/3mdad_f1.json --wall-seconds 86400
python tools/reeval_rules.py --target 3mdad --fold 2 --out analysis/reeval/3mdad_f2.json --wall-seconds 86400
python tools/reeval_rules.py --target dmd --fold 0 --out analysis/reeval/dmd_f0.json --wall-seconds 86400
python tools/reeval_rules.py --target dmd --fold 1 --out analysis/reeval/dmd_f1.json --wall-seconds 86400
python tools/reeval_rules.py --target dmd --fold 2 --out analysis/reeval/dmd_f2.json --wall-seconds 86400
