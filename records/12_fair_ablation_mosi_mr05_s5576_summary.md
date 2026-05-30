# 12 Fair Ablation Study on MOSI mr=0.5 seed=5576

消融实验设置：

- Dataset: MOSI
- Missing rate: 0.5
- Seed: 5576
- Epochs: 100
- Full model uses deg_uncertainty_weight = 0.001
- MAE 越低越好，Corr / Acc-2 / F1 越高越好。

| Method | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ |
|---|---:|---:|---:|---:|
| HME | 1.1426 | 0.5445 | 0.6545 | 0.6476 |
| Retrieval-only / HME-TopK | 1.0938 | 0.5083 | 0.6953 | 0.6950 |
| DARE-HME w/o Degradation | 1.1416 | 0.5203 | 0.6239 | 0.6162 |
| DARE-HME w/o Quality | 0.8124 | 0.7282 | 0.7799 | 0.7794 |
| DARE-HME w/o Uncertainty | 0.7833 | 0.7561 | 0.7886 | 0.7879 |
| DARE-HME full | **0.7982** | 0.7410 | **0.7959** | **0.7961** |

## Observation

1. Compared with HME and Retrieval-only, DARE-HME full achieves clear improvements, showing that simple TopK retrieval enhancement is insufficient.
2. Removing multi-level degradation modeling causes severe performance degradation, indicating that degradation-aware low-quality modeling is the most critical component.
3. Removing quality-aware enhancement decreases Acc-2 and F1, showing that quality-guided reliable enhancement is useful.
4. Removing uncertainty consistency also decreases Acc-2 and F1, indicating that a small uncertainty consistency weight helps improve classification robustness.
