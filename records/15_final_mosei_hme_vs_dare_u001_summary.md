# 15 Final MOSEI HME vs DARE-HME u=0.001 Summary

MOSEI 泛化验证表。DARE-HME 使用最终配置 `deg_uncertainty_weight=0.001`。

| Dataset | MR | Model | Seed | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ |
|---|---:|---|---:|---:|---:|---:|---:|
| MOSEI | 0.2 | DARE-HME u=0.001 | 5576 | 0.5495 | 0.7560 | 0.8083 | 0.8139 |
| MOSEI | 0.2 | HME | 5576 | 0.6011 | 0.6755 | 0.7839 | 0.7859 |
| MOSEI | 0.5 | DARE-HME u=0.001 | 5576 | 0.5663 | 0.7344 | 0.8075 | 0.8105 |
| MOSEI | 0.5 | HME | 5576 | 0.6833 | 0.5402 | 0.7334 | 0.7312 |

## Improvement

| MR | MAE reduction | Corr gain | Acc-2 gain | F1 gain |
|---:|---:|---:|---:|---:|
| 0.2 | 0.0516 | 0.0805 | +2.44 pp | +2.80 pp |
| 0.5 | 0.1170 | 0.1942 | +7.41 pp | +7.93 pp |
