#!/bin/bash

cd /root/private_data/HME/HME

echo "============================================================"
echo "Start pilot multi-seed check: MOSI seed=2222 epochs=20"
echo "Missing rates: 0.2 0.3 0.5"
echo "Models: HME / HME-TopK / DARE-HME"
echo "============================================================"

RATES=(0.2 0.3 0.5)

for MR in "${RATES[@]}"
do
  echo ""
  echo "############################################################"
  echo "Current missing_rate = ${MR}, seed = 2222"
  echo "############################################################"

  echo ""
  echo "==================== HME baseline, MR=${MR} ===================="
  python tools/run_experiment.py \
    --model hme \
    --dataset mosi \
    --missing_rate ${MR} \
    --seed 2222 \
    --epochs 20 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --tag pilot \
    --note "pilot multi-seed check on MOSI seed=2222"

  echo ""
  echo "==================== HME-TopK, MR=${MR} ===================="
  python tools/run_experiment.py \
    --model topk \
    --dataset mosi \
    --missing_rate ${MR} \
    --seed 2222 \
    --epochs 20 \
    --top_k 3 \
    --temperature 0.1 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --tag pilot \
    --note "pilot multi-seed check on MOSI seed=2222"

  echo ""
  echo "==================== DARE-HME, MR=${MR} ===================="
  python tools/run_experiment.py \
    --model dare \
    --dataset mosi \
    --missing_rate ${MR} \
    --seed 2222 \
    --epochs 20 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --tag pilot \
    --note "pilot multi-seed check on MOSI seed=2222"

done

echo ""
echo "============================================================"
echo "Seed=2222 pilot experiments finished."
echo "Check records/pretty_logs/00_all_runs_pretty.log"
echo "============================================================"
