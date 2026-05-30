# Server Run Status

Date: 2026-05-27

## Environment
Python 3.8
torch 1.12.1+cu113
transformers 4.30.2
CUDA available: True
GPU: RTX 4090

## Verified Runs

### HME baseline, MOSI, missing_rate=0.0, n_epochs=1
train_loss: 2.3406
valid_loss: 2.4169
test_acc2: 0.7857
mae: 1.1841
corr: 0.6652

### HME baseline, MOSI, missing_rate=0.2, n_epochs=1
train_loss: 2.3681
valid_loss: 2.4292
test_acc2: 0.7362
mae: 1.2553
corr: 0.5903

### DARE-HME, MOSI, missing_rate=0.2, n_epochs=1
train_loss: 2.3694
valid_loss: 2.4078
test_acc2: 0.7843
mae: 1.2110
corr: 0.6789

## Conclusion
Baseline and DARE-HME can run normally on the server after aligning the environment with local versions.
Previous NaN issue was mainly caused by environment mismatch.
