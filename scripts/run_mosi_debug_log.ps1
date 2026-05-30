# ================================
# HME-TopK MOSI Debug Run
# ================================

# 进入 HME 项目根目录
cd F:\Code\Study\MMSA_Week4\codes\HME

# 激活 HME 环境
conda activate hme38

# 运行 HME-TopK 1 epoch 调试
python HME_MSA/HME_main_topk.py `
  --dataset mosi `
  --n_epochs 1 `
  --train_batch_size 16 `
  --dev_batch_size 16 `
  --test_batch_size 16 `
  --top_k 3 `
  --top_temperature 0.1