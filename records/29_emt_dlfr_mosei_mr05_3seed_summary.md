# 29 EMT-DLFR on MOSEI MR=0.5 3-Seed Summary

## Setting

- Dataset: CMU-MOSEI
- Missing rate: MR=0.5
- Model: EMT-DLFR-adapted
- Seeds: 1111 / 2222 / 5576
- Metrics are converted from percentage-style CSV values to decimal values.

## Per-seed Results

| Seed | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | MAE ↓ | Corr ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 1111 | 0.7742 | 0.7750 | 0.7788 | 0.7741 | 0.6837 | 0.5633 |
| 2222 | 0.7641 | 0.7660 | 0.7708 | 0.7674 | 0.6820 | 0.5620 |
| 5576 | 0.6922 | 0.7058 | 0.7356 | 0.7391 | 0.6837 | 0.5478 |

## Mean ± Std

| Model | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | MAE ↓ | Corr ↑ |
|---|---:|---:|---:|---:|---:|---:|
| EMT-DLFR | 0.7435±0.0365 | 0.7489±0.0307 | 0.7617±0.0188 | 0.7602±0.0152 | 0.6831±0.0008 | 0.5577±0.0070 |

## Paper-use Row

For the main table, use Has0 Acc-2/F1 together with MAE and Corr:

| Model | Acc-2 | F1 | MAE ↓ | Corr ↑ |
|---|---:|---:|---:|---:|
| EMT-DLFR | 0.7435±0.0365 | 0.7489±0.0307 | 0.6831±0.0008 | 0.5577±0.0070 |

## Note

This result is used as an external adapted baseline for the ICASSP-style first paper.
