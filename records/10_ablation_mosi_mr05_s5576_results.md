# 10 Ablation MOSI mr=0.5 seed=5576 Results

这里记录 DARE-HME 三个核心模块的单 seed 消融结果。

| run_id | ablation | status | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ | log_file |
|---|---|---|---:|---:|---:|---:|---|
| ablation_mosi_wo_quality_mr05_s5576_e100 | wo_quality | finished | 0.8256 | 0.7347 | 0.7813 | 0.7819 | logs/ablation/ablation_mosi_wo_quality_mr05_s5576_e100.log |
| ablation_mosi_wo_uncertainty_mr05_s5576_e100 | wo_uncertainty | finished | 0.7833 | 0.7561 | 0.7886 | 0.7879 | logs/ablation/ablation_mosi_wo_uncertainty_mr05_s5576_e100.log |
| ablation_mosi_wo_degradation_mr05_s5576_e100 | wo_degradation | finished | 1.1279 | 0.5032 | 0.6501 | 0.65 | logs/ablation/ablation_mosi_wo_degradation_mr05_s5576_e100.log |
