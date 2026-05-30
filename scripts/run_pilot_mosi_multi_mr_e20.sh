#!/bin/bash

cd /root/private_data/HME/HME

echo "============================================================"
echo "Start pilot experiment: MOSI multi missing rates, epochs=20"
echo "Missing rates: 0.0 0.1 0.3 0.5"
echo "Models: HME / HME-TopK / DARE-HME"
echo "============================================================"

RATES=(0.0 0.1 0.3 0.5)

for MR in "${RATES[@]}"
do
  echo ""
  echo "############################################################"
  echo "Current missing_rate = ${MR}"
  echo "############################################################"

  echo ""
  echo "==================== HME baseline, MR=${MR} ===================="
  python tools/run_experiment.py \
    --model hme \
    --dataset mosi \
    --missing_rate ${MR} \
    --seed 5576 \
    --epochs 20 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --tag pilot \
    --note "pilot comparison on MOSI multi missing rates seed=5576"

  echo ""
  echo "==================== HME-TopK, MR=${MR} ===================="
  python tools/run_experiment.py \
    --model topk \
    --dataset mosi \
    --missing_rate ${MR} \
    --seed 5576 \
    --epochs 20 \
    --top_k 3 \
    --temperature 0.1 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --tag pilot \
    --note "pilot comparison on MOSI multi missing rates seed=5576"

  echo ""
  echo "==================== DARE-HME, MR=${MR} ===================="
  python tools/run_experiment.py \
    --model dare \
    --dataset mosi \
    --missing_rate ${MR} \
    --seed 5576 \
    --epochs 20 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --tag pilot \
    --note "pilot comparison on MOSI multi missing rates seed=5576"

done

echo ""
echo "============================================================"
echo "All MOSI multi missing-rate pilot experiments finished."
echo "Check records/pretty_logs/00_all_runs_pretty.log"
echo "============================================================"
