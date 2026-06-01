# 27 Low-quality Scenario Stress Test Decision

This experiment evaluates modality-specific severe degradation on MOSI mr=0.5 seed=5576.

## Important Note

This experiment is an exploratory stress test and will not be used as the main story of the first ICASSP-style paper.

The first paper focuses on random multi-source low-quality / missing modalities, where text, audio, and vision may be randomly degraded with a given probability.

## Observation

The corrected lq_scenario setting is effective. However, DARE-HME does not consistently outperform HME under fixed modality-specific degradation.

In particular:

- DARE-HME slightly improves classification under vision and audio_vision degradation.
- DARE-HME does not outperform HME under audio degradation.
- DARE-HME performs worse under text degradation, showing that text remains the dominant sentiment carrier in MOSI.

## Decision

Do not use this table as a main paper result.

The ICASSP-style paper should focus on:

1. MOSI multi-missing-rate results: MR=0.2 / 0.3 / 0.5.
2. MOSEI generalization results.
3. External baselines: TFR-Net, LNLN, EMT-DLFR, and reported recent methods.
4. Ablation study.
5. Sensitivity analysis if space allows.

This stress-test result can be kept as internal exploration or thesis limitation analysis.
