#!/bin/bash
# Watchdog: automatically resumes DART training if it crashes/gets killed.
# Keeps restarting until 100K steps are reached.
# Usage: bash train_watchdog.sh

TARGET_STEPS=100000
CHECKPOINT="checkpoints/dart_small_latest.pt"

while true; do
    # Check if target checkpoint exists (training already completed)
    if [ -f "checkpoints/dart_small_step${TARGET_STEPS}.safetensors" ]; then
        echo "[watchdog] Training complete (step ${TARGET_STEPS} checkpoint found). Exiting."
        break
    fi

    # Build resume flag if checkpoint exists
    RESUME_FLAG=""
    if [ -f "$CHECKPOINT" ]; then
        RESUME_FLAG="--resume $CHECKPOINT"
        echo "[watchdog] Resuming from $CHECKPOINT"
    else
        echo "[watchdog] Starting fresh training"
    fi

    echo "[watchdog] Launching training at $(date)"
    PYTHONUNBUFFERED=1 python train/train.py \
        --size small \
        --dataset cifar10 \
        --num-steps 4 \
        --epochs 100000 \
        --steps $TARGET_STEPS \
        --sample-every 5000 \
        --save-every 5000 \
        --batch-size 8 \
        --workers 0 \
        $RESUME_FLAG

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[watchdog] Training finished successfully."
        break
    fi

    echo "[watchdog] Training exited with code $EXIT_CODE at $(date). Restarting in 10s..."
    sleep 10
done
