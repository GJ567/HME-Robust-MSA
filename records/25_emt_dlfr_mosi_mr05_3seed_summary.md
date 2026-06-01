# 25 EMT-DLFR-adapted MOSI mr=0.5 3-Seed Summary

该实验使用 EMT-DLFR 官方代码，并适配当前 HME 工程中的 compatible MOSI 数据，因此暂记为 EMT-DLFR-adapted pilot baseline。

## Per-seed Results

| Seed | MAE ↓ | Corr ↑ | Has0 Acc-2 ↑ | Has0 F1 ↑ | Non0 Acc-2 ↑ | Non0 F1 ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 1111 | 1.1197 | 0.5099 | 0.7070 | 0.7078 | 0.7119 | 0.7136 |
| 2222 | 1.1140 | 0.4860 | 0.7026 | 0.7028 | 0.7104 | 0.7114 |
| 5576 | 1.1197 | 0.4921 | 0.7114 | 0.7120 | 0.7165 | 0.7179 |

## Mean ± Std

| Method | MAE ↓ | Corr ↑ | Has0 Acc-2 ↑ | Has0 F1 ↑ | Non0 Acc-2 ↑ | Non0 F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| EMT-DLFR-adapted | 1.1178±0.0027 | 0.4960±0.0101 | 0.7070±0.0036 | 0.7075±0.0038 | 0.7129±0.0026 | 0.7143±0.0027 |

## Note

EMT-DLFR-adapted is a recent external baseline. Since the data format is converted from the current HME pipeline, this result is treated as an adapted pilot comparison rather than a strict official reproduction.