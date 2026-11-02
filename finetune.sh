#!/usr/bin/env bash
# Example GR00T N1.7 fine-tuning launches for the UNITREE_G1_SONIC embodiment.
# Set the paths below (or export them) before running; each block is an
# independent example (8 GPUs, 1 GPU, 4 GPUs).
BASE_MODEL_PATH="${BASE_MODEL_PATH:?set BASE_MODEL_PATH to a local GR00T-N1.7-3B snapshot}"
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to a LeRobot v2.1 dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the training output root}"

uv run torchrun --nproc_per_node=8 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag UNITREE_G1_SONIC \
    --modality-config-path gr00t/configs/data/embodiment_configs.py \
    --num-gpus 8 \
    --output-dir "$OUTPUT_ROOT/ae_push_squat_close" \
    --save-total-limit 5 \
    --save-steps 5000 \
    --max-steps 20000 \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4


CUDA_VISIBLE_DEVICES=2 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag UNITREE_G1_SONIC \
    --modality-config-path gr00t/configs/data/embodiment_configs.py \
    --num-gpus 1 \
    --output-dir "$OUTPUT_ROOT/ae_merge_test" \
    --save-total-limit 5 \
    --save-steps 5000 \
    --max-steps 20000 \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4

 CUDA_VISIBLE_DEVICES=4,5,6,7 uv run torchrun --nproc_per_node=4 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag UNITREE_G1_SONIC \
    --modality-config-path gr00t/configs/data/embodiment_configs.py \
    --num-gpus 4 \
    --output-dir "$OUTPUT_ROOT/ae_push_squat_close_image" \
    --save-total-limit 5 \
    --save-steps 20000 \
    --max-steps 100000 \
    --global-batch-size 128 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4