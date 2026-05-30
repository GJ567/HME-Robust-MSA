# 10 Ablation MOSI mr=0.5 seed=5576 Summary

消融实验汇总。MAE 越低越好，Corr / Acc-2 / F1 越高越好。

| Method | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ |
|---|---:|---:|---:|---:|
| HME | 1.1426 | 0.5445 | 0.6545 | 0.6476 |
| Retrieval-only / HME-TopK | 1.0938 | 0.5083 | 0.6953 | 0.6950 |
| DARE-HME full | 0.8122 | 0.7313 | 0.7828 | 0.7832 |
| DARE-HME w/o Quality | 0.8256 | 0.7347 | 0.7813 | 0.7819 |
| DARE-HME w/o Uncertainty | 0.7833 | 0.7561 | 0.7886 | 0.7879 |
| DARE-HME w/o Degradation | 1.1279 | 0.5032 | 0.6501 | 0.6500 |
