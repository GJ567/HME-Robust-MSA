# 07 Pilot MOSI 3-Seed Improvement

这里统计 DARE-HME 相比 HME / HME-TopK 的平均提升。MAE reduction 越大越好，其余 gain 越大越好。

| missing_rate | comparison | mae_reduction | corr_gain | acc2_gain | acc2_gain_percent_point | f1_gain | f1_gain_percent_point |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | DARE-HME vs HME | 0.1061 | 0.074 | 0.0433 | 4.33 | 0.0425 | 4.25 |
| 0.2 | DARE-HME vs HME-TopK | 0.1062 | 0.0714 | 0.035 | 3.5 | 0.0345 | 3.45 |
| 0.3 | DARE-HME vs HME | 0.1639 | 0.1118 | 0.0607 | 6.07 | 0.0612 | 6.12 |
| 0.3 | DARE-HME vs HME-TopK | 0.1514 | 0.0963 | 0.0554 | 5.54 | 0.057 | 5.7 |
| 0.5 | DARE-HME vs HME | 0.3087 | 0.2231 | 0.1317 | 13.17 | 0.1378 | 13.78 |
| 0.5 | DARE-HME vs HME-TopK | 0.2965 | 0.2108 | 0.1098 | 10.98 | 0.1153 | 11.53 |
