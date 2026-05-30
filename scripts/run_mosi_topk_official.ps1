# ================================
# HME-TopK MOSI Official Run
# ================================

cd F:\Code\Study\MMSA_Week4\codes\HME

conda activate hme38

python HME_MSA/HME_main_topk.py `
  --dataset mosi `
  --n_epochs 100 `
  --train_batch_size 256 `
  --dev_batch_size 128 `
  --test_batch_size 128 `
  --top_k 3 `
  --top_temperature 0.1