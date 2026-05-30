# ================================
# DARE-HME MOSI Official Ablation Runs
# ================================

cd F:\Code\Study\MMSA_Week4\codes\HME

conda activate hme38

# 1. DARE-HME Full
python HME_MSA/HME_main_dare.py `
  --dataset mosi `
  --n_epochs 100 `
  --train_batch_size 256 `
  --dev_batch_size 128 `
  --test_batch_size 128 `
  --top_k 3 `
  --top_temperature 0.1 `
  --use_quality_weight True `
  --deg_uncertainty_weight 0.01

# 2. w/o Quality Weight
python HME_MSA/HME_main_dare.py `
  --dataset mosi `
  --n_epochs 100 `
  --train_batch_size 256 `
  --dev_batch_size 128 `
  --test_batch_size 128 `
  --top_k 3 `
  --top_temperature 0.1 `
  --use_quality_weight False `
  --deg_uncertainty_weight 0.01

# 3. w/o Uncertainty Loss
python HME_MSA/HME_main_dare.py `
  --dataset mosi `
  --n_epochs 100 `
  --train_batch_size 256 `
  --dev_batch_size 128 `
  --test_batch_size 128 `
  --top_k 3 `
  --top_temperature 0.1 `
  --use_quality_weight True `
  --deg_uncertainty_weight 0.0

# 4. w/o Quality Weight & w/o Uncertainty Loss
python HME_MSA/HME_main_dare.py `
  --dataset mosi `
  --n_epochs 100 `
  --train_batch_size 256 `
  --dev_batch_size 128 `
  --test_batch_size 128 `
  --top_k 3 `
  --top_temperature 0.1 `
  --use_quality_weight False `
  --deg_uncertainty_weight 0.0