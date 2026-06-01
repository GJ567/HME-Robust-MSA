# 28 ICASSP-style First Paper Table Plan

## Core Story

The first paper focuses on random multi-source low-quality / missing modalities.

We do not claim that DARE-HME outperforms HME under all fixed single-modality degradation scenarios.

## Table 1: Main Results under MR=0.5

Format: left CMU-MOSI, right CMU-MOSEI.

Metrics:

- Acc-2
- F1
- MAE
- Corr

Methods:

1. HME
2. TFR-Net
3. EMT-DLFR
4. LNLN
5. UMDF reported
6. MPLMM reported
7. P-RMF reported
8. DARE-HME

## Table 2: Robustness under Different Missing Rates

Only compare HME and DARE-HME.

MOSI:

- MR=0.2
- MR=0.3
- MR=0.5

MOSEI:

- MR=0.2
- MR=0.5

Use Acc-2/F1 in the compact table. Mention MAE/Corr trends in text if space is limited.

## Table 3: Ablation Study

Use MOSI MR=0.5 seed=5576.

Rows:

- HME
- Retrieval-only / HME-TopK
- w/o Degradation
- w/o Quality
- w/o Uncertainty
- DARE-HME full

## Not Used in Main Paper

The modality-specific stress test with audio, vision, text and audio_vision degradation will not be used in the main paper. It is kept as internal exploratory analysis.
