#!/bin/bash

cd /root/private_data/HME/HME

echo "============================================================"
echo "Official MOSI mr=0.3 seeds=5576/1111/2222 epochs=100"
echo "Models: HME / DARE-HME"
echo "============================================================"

SEEDS=(5576 1111 2222)

for SEED in "${SEEDS[@]}"
do
  echo ""
  echo "############################################################"
  echo "Current seed = ${SEED}, missing_rate = 0.3"
  echo "############################################################"

  echo ""
  echo "==================== HME baseline ===================="
  python tools/run_experiment.py \
    --model hme \
    --dataset mosi \
    --missing_rate 0.3 \
    --seed ${SEED} \
    --epochs 100 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --tag official \
    --note "official MOSI mr=0.3 multi-seed check"

  echo ""
  echo "==================== DARE-HME ===================="
  python tools/run_experiment.py \
    --model dare \
    --dataset mosi \
    --missing_rate 0.3 \
    --seed ${SEED} \
    --epochs 100 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --tag official \
    --note "official MOSI mr=0.3 multi-seed check"

done

echo ""
echo "============================================================"
echo "Official MOSI mr=0.3 finished."
echo "============================================================"
