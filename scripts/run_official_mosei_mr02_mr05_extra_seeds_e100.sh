#!/bin/bash

cd /root/private_data/HME/HME

echo "============================================================"
echo "Official MOSEI mr=0.2/0.5 extra seeds=1111/2222 epochs=100"
echo "Models: HME / DARE-HME"
echo "Purpose: complete MOSEI multi-seed generalization check"
echo "============================================================"

SEEDS=(1111 2222)
MISSING_RATES=(0.2 0.5)

for MR in "${MISSING_RATES[@]}"
do
  for SEED in "${SEEDS[@]}"
  do
    echo ""
    echo "############################################################"
    echo "MOSEI | missing_rate=${MR} | seed=${SEED}"
    echo "############################################################"

    echo ""
    echo "==================== HME baseline ===================="
    python tools/run_experiment.py \
      --model hme \
      --dataset mosei \
      --missing_rate ${MR} \
      --seed ${SEED} \
      --epochs 100 \
      --train_batch_size 8 \
      --dev_batch_size 8 \
      --test_batch_size 8 \
      --tag official \
      --note "official MOSEI mr=${MR} seed=${SEED} generalization multi-seed check"

    echo ""
    echo "==================== DARE-HME ===================="
    python tools/run_experiment.py \
      --model dare \
      --dataset mosei \
      --missing_rate ${MR} \
      --seed ${SEED} \
      --epochs 100 \
      --train_batch_size 8 \
      --dev_batch_size 8 \
      --test_batch_size 8 \
      --tag official \
      --note "official MOSEI mr=${MR} seed=${SEED} generalization multi-seed check"

  done
done

echo ""
echo "============================================================"
echo "Official MOSEI extra seeds finished."
echo "============================================================"
