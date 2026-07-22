# Pipeline validation

## One-batch overfitting test

- Status: passed
- Run name: debug-one-batch-overfit-s1
- WandB run: <https://wandb.ai/varvaravolodicheva-hse-university/voice-antispoofing-lcnn>
- Epochs: 200
- Batch size: 8
- Dataset:
  - 4 bonafide samples
  - 4 spoof samples
  - the same fixed batch was used for train, dev and eval
- Final metrics:
  - train loss: 0.0143
  - train accuracy: 1.0000
  - dev loss: 0.0023
  - dev accuracy: 1.0000
  - eval loss: 0.0023
  - eval accuracy: 1.0000
  - eval EER: 0.0%
- Checkpoint:
  - saved/debug-one-batch-overfit-s1/checkpoint-epoch200.pth
- Conclusion:
  - The model successfully overfitted a fixed balanced batch.
  - Forward pass, loss calculation, backward pass,
    optimizer update and EER calculation work correctly.
  ### Command
  ```bash
python train.py \
  datasets=asvspoof_one_batch \
  trainer.n_epochs=200 \
  trainer.epoch_len=1 \
  trainer.log_step=1 \
  trainer.save_period=200 \
  dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  dataloader.persistent_workers=false \
  writer.run_name="debug-one-batch-overfit-s1"

## Smoke test

- Status: core pipeline passed, inference pending
- Date: 2026-07-22
- Git commit: 1f9e89706bbdf72f074868755348f9da5d617910
- WandB run: short-pipeline-test
- Train examples: 16
- Dev examples: 16
- Eval examples: 64
- Optimizer steps: 2
- Result:
  - Dataset/DataLoader: passed
  - Forward/loss/backward: passed
  - Optimizer step: passed
  - Evaluation: passed
  - WandB: passed
  - Checkpoint: not tested
  - Inference: not tested
  - CSV export: not tested
