# 11 Uncertainty Weight Sensitivity on MOSI mr=0.5 seed=5576

这里比较不同 `deg_uncertainty_weight` 对 DARE-HME 的影响。

| weight | setting | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ | log_file |
|---:|---|---:|---:|---:|---:|---|
| 0.0 | w/o Uncertainty | 0.7833 | 0.7561 | 0.7886 | 0.7879 | records/10_ablation_mosi_mr05_s5576_summary.md |
| 0.001 | sensitivity | 0.7982 | 0.7410 | 0.7959 | 0.7961 | logs/uncertainty_sensitivity/mosi_mr05_s5576_e100_u001.log |
| 0.003 | sensitivity | 0.8068 | 0.7288 | 0.7930 | 0.7930 | logs/uncertainty_sensitivity/mosi_mr05_s5576_e100_u003.log |
| 0.005 | sensitivity | 0.7998 | 0.7415 | 0.7857 | 0.7861 | logs/uncertainty_sensitivity/mosi_mr05_s5576_e100_u005.log |
| 0.01 | DARE-HME full original | 0.8122 | 0.7313 | 0.7828 | 0.7832 | records/04_dare_hme_results.csv |
