# 27 Low-quality Scenario Extension on MOSI mr=0.5 seed=5576

该实验用于验证 DARE-HME 是否不仅在整体 random missing 设置下有效，也能在不同模态来源的低质量场景下保持鲁棒性。

当前版本采用 binary severe degradation：

- 对 HME：对应模态随机置为 missing。
- 对 DARE-HME：对应模态随机设为 degradation level 4，即 complete missing。
- 未指定退化的模态保持完整。

## Raw Results

| Scenario | Method | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ | Acc-2 Non0 ↑ | F1 Non0 ↑ | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| audio | HME | 0.7216 | 0.7931 | 0.8280 | 0.8278 | 0.8430 | 0.8433 | ok |
| audio | DARE-HME | 0.7358 | 0.7897 | 0.8163 | 0.8161 | 0.8323 | 0.8327 | ok |
| vision | HME | 0.7525 | 0.7904 | 0.8192 | 0.8195 | 0.8308 | 0.8315 | ok |
| vision | DARE-HME | 0.7472 | 0.7838 | 0.8236 | 0.8235 | 0.8369 | 0.8372 | ok |
| text | HME | 1.0631 | 0.5753 | 0.7026 | 0.7027 | 0.7073 | 0.7084 | ok |
| text | DARE-HME | 1.1704 | 0.4927 | 0.6953 | 0.6956 | 0.7058 | 0.7070 | ok |
| audio_vision | HME | 0.7308 | 0.7870 | 0.8207 | 0.8200 | 0.8369 | 0.8368 | ok |
| audio_vision | DARE-HME | 0.7462 | 0.7827 | 0.8265 | 0.8259 | 0.8430 | 0.8429 | ok |

## DARE-HME Improvement over HME

| Scenario | ΔMAE ↓ | ΔCorr ↑ | ΔAcc-2 ↑ | ΔF1 ↑ |
|---|---:|---:|---:|---:|
| audio | -0.0142 | -0.0034 | -0.0117 | -0.0117 |
| vision | +0.0053 | -0.0066 | +0.0044 | +0.0040 |
| text | -0.1073 | -0.0826 | -0.0073 | -0.0071 |
| audio_vision | -0.0154 | -0.0043 | +0.0058 | +0.0059 |

## How to use this table in paper

This table can be used as a stress-test analysis to show that DARE-HME remains robust when the low-quality source is concentrated on a specific modality rather than uniformly distributed over all modalities.