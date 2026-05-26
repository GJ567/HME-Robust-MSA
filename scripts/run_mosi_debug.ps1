cd F:\Code\Study\MMSA_week4\codes\HME

conda activate hme38

$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

python .\HME_MSA\HME_main.py `
  --dataset='mosi' `
  --learning_rate=2e-5 `
  --d_l=64 `
  --missing_rate=0.2 `
  --layers=1 `
  --hyper_depth=1 `
  --latent_layers=1 `
  --latent_dim=64 `
  --num_latents=2 `
  --train_batch_size=8 `
  --dev_batch_size=8 `
  --test_batch_size=8 `
  --n_epochs=2