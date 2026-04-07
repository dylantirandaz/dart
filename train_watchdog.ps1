# Watchdog: automatically resumes DART training if it crashes/gets killed.
# Keeps restarting until target steps are reached.
# Usage: powershell -ExecutionPolicy Bypass -File train_watchdog.ps1

$TARGET_STEPS = 100000
$CHECKPOINT = "checkpoints\dart_small_latest.pt"
$env:PYTHONUNBUFFERED = "1"

while ($true) {
    # Check actual step in latest checkpoint
    if (Test-Path $CHECKPOINT) {
        [int]$step = & python -c "import torch; ckpt=torch.load('$CHECKPOINT',map_location='cpu',weights_only=False); print(ckpt['global_step'])" 2>$null
        if ($step -ge $TARGET_STEPS) {
            Write-Host "[watchdog] Training complete (checkpoint at step $step). Exiting."
            break
        }
        Write-Host "[watchdog] Resuming from step $step"
        Write-Host "[watchdog] Launching training at $(Get-Date)"
        & python train/train.py --size small --dataset cifar10 --num-steps 4 --epochs 100000 --steps $TARGET_STEPS --sample-every 5000 --save-every 5000 --batch-size 8 --workers 0 --resume $CHECKPOINT
    } else {
        Write-Host "[watchdog] Starting fresh training"
        Write-Host "[watchdog] Launching training at $(Get-Date)"
        & python train/train.py --size small --dataset cifar10 --num-steps 4 --epochs 100000 --steps $TARGET_STEPS --sample-every 5000 --save-every 5000 --batch-size 8 --workers 0
    }

    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        Write-Host "[watchdog] Training finished successfully."
        break
    }

    Write-Host "[watchdog] Training exited with code $exitCode at $(Get-Date). Restarting in 10s..."
    Start-Sleep -Seconds 10
}
