#!/bin/bash

cd /root/private_data/HME/HME

echo "============================================================"
echo "Official check: MOSI mr=0.5 seed=5576 epochs=100"
echo "Models: HME / HME-TopK / DARE-HME"
echo "============================================================"

echo ""
echo "==================== 1. HME baseline ===================="
python tools/run_experiment.py \
  --model hme \
  --dataset mosi \
  --missing_rate 0.5 \
  --seed 5576 \
  --epochs 100 \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --tag official \
  --note "official check on MOSI mr=0.5 seed=5576 epochs=100"

echo ""
echo "==================== 2. HME-TopK ===================="
python tools/run_experiment.py \
  --model topk \
  --dataset mosi \
  --missing_rate 0.5 \
  --seed 5576 \
  --epochs 100 \
  --top_k 3 \
  --temperature 0.1 \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --tag official \
  --note "official check on MOSI mr=0.5 seed=5576 epochs=100"

echo ""
echo "==================== 3. DARE-HME ===================="
python tools/run_experiment.py \
  --model dare \
  --dataset mosi \
  --missing_rate 0.5 \
  --seed 5576 \
  --epochs 100 \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --tag official \
  --note "official check on MOSI mr=0.5 seed=5576 epochs=100"

echo ""
echo "============================================================"
echo "Official check finished."
echo "Check records/pretty_logs/00_all_runs_pretty.log"
echo "============================================================"
