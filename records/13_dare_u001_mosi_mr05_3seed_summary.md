# 13 DARE-HME u=0.001 MOSI mr=0.5 3-seed Summary

新版 DARE-HME full 使用 `deg_uncertainty_weight=0.001`。

| seed | status | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ | log_file |
|---:|---|---:|---:|---:|---:|---|
| 5576 | finished | 0.7982 | 0.7410 | 0.7959 | 0.7961 | logs/uncertainty_sensitivity/mosi_mr05_s5576_e100_u001.log |
| 1111 | finished | 0.7922 | 0.7393 | 0.8017 | 0.8023 | logs/official_u001/dare_u001_mosi_mr05_s1111_e100.log |
| 2222 | finished | 0.7821 | 0.7568 | 0.7755 | 0.7747 | logs/official_u001/dare_u001_mosi_mr05_s2222_e100.log |

## Mean ± Std

| Model | Seeds | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ |
|---|---|---:|---:|---:|---:|
| DARE-HME u=0.001 | 1111/2222/5576 | 0.7908±0.0081 | 0.7457±0.0097 | 0.7910±0.0138 | 0.7910±0.0145 |
