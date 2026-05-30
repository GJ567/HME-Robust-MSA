#!/bin/bash

cd /root/private_data/HME/HME

echo "============================================================"
echo "Official MOSI mr=0.5 seeds=1111/2222 epochs=100"
echo "Models: HME / HME-TopK / DARE-HME"
echo "============================================================"

SEEDS=(1111 2222)

for SEED in "${SEEDS[@]}"
do
  echo ""
  echo "############################################################"
  echo "Current seed = ${SEED}, missing_rate = 0.5"
  echo "############################################################"

  echo ""
  echo "==================== HME baseline ===================="
  python tools/run_experiment.py \
    --model hme \
    --dataset mosi \
    --missing_rate 0.5 \
    --seed ${SEED} \
    --epochs 100 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --tag official \
    --note "official MOSI mr=0.5 multi-seed check"

  echo ""
  echo "==================== HME-TopK ===================="
  python tools/run_experiment.py \
    --model topk \
    --dataset mosi \
    --missing_rate 0.5 \
    --seed ${SEED} \
    --epochs 100 \
    --top_k 3 \
    --temperature 0.1 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --tag official \
    --note "official MOSI mr=0.5 multi-seed check"

  echo ""
  echo "==================== DARE-HME ===================="
  python tools/run_experiment.py \
    --model dare \
    --dataset mosi \
    --missing_rate 0.5 \
    --seed ${SEED} \
    --epochs 100 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --tag official \
    --note "official MOSI mr=0.5 multi-seed check"

done

echo ""
echo "============================================================"
echo "Official MOSI mr=0.5 seeds=1111/2222 finished."
echo "============================================================"
