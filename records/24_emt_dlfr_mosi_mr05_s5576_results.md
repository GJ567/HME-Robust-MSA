# 24 EMT-DLFR-adapted MOSI mr=0.5 seed=5576 Results

该实验使用 EMT-DLFR 官方代码，并适配当前 HME 工程中的 compatible MOSI 数据，因此暂记为 EMT-DLFR-adapted pilot baseline。

## Setting

| Item | Value |
|---|---|
| Dataset | MOSI |
| Missing rate | 0.5 |
| Seed | 5576 |
| Model | EMT-DLFR |
| Data format | HME-compatible converted MOSI |
| Note | adapted pilot baseline, not strict official reproduction |

## Test Results

| Method | MAE ↓ | Corr ↑ | Has0 Acc-2 ↑ | Has0 F1 ↑ | Non0 Acc-2 ↑ | Non0 F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| EMT-DLFR-adapted | 1.1197 | 0.4921 | 0.7114 | 0.7120 | 0.7165 | 0.7179 |

## Observation

EMT-DLFR-adapted improves classification performance compared with HME, LNLN-adapted and TFR-Net-adapted under MOSI mr=0.5. However, DARE-HME still achieves clear improvements across MAE, Corr, Acc-2 and F1.
