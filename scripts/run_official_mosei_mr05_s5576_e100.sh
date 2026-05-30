#!/bin/bash

cd /root/private_data/HME/HME

echo "============================================================"
echo "Official MOSEI mr=0.5 seed=5576 epochs=100"
echo "Models: HME / DARE-HME"
echo "============================================================"

echo ""
echo "==================== HME baseline ===================="
python tools/run_experiment.py \
  --model hme \
  --dataset mosei \
  --missing_rate 0.5 \
  --seed 5576 \
  --epochs 100 \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --tag official \
  --note "official MOSEI mr=0.5 seed=5576 generalization check"

echo ""
echo "==================== DARE-HME ===================="
python tools/run_experiment.py \
  --model dare \
  --dataset mosei \
  --missing_rate 0.5 \
  --seed 5576 \
  --epochs 100 \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --tag official \
  --note "official MOSEI mr=0.5 seed=5576 generalization check"

echo ""
echo "============================================================"
echo "Official MOSEI mr=0.5 finished."
echo "============================================================"
