#!/bin/bash

cd /root/private_data/HME/HME

echo "============================================================"
echo "Official DARE-HME u=0.001 on MOSI mr=0.2 / 0.3"
echo "Seeds: 5576 / 1111 / 2222"
echo "Epochs: 100"
echo "Purpose: replace old DARE-HME full weight=0.01"
echo "============================================================"

SEEDS=(5576 1111 2222)
MRS=(0.2 0.3)

for MR in "${MRS[@]}"
do
  if [ "$MR" = "0.2" ]; then
    MRTAG="mr02"
  elif [ "$MR" = "0.3" ]; then
    MRTAG="mr03"
  else
    MRTAG="mrxx"
  fi

  for SEED in "${SEEDS[@]}"
  do
    echo ""
    echo "############################################################"
    echo "Running DARE-HME u=0.001, MOSI ${MRTAG}, seed=${SEED}"
    echo "############################################################"

    LOG_FILE="logs/official_u001/dare_u001_mosi_${MRTAG}_s${SEED}_e100.log"

    script -q -c "/opt/conda/envs/hme38/bin/python HME_MSA/HME_main_dare.py \
      --dataset mosi \
      --learning_rate 2e-5 \
      --d_l 64 \
      --missing_rate ${MR} \
      --layers 1 \
      --hyper_depth 1 \
      --latent_layers 1 \
      --latent_dim 64 \
      --num_latents 2 \
      --train_batch_size 8 \
      --dev_batch_size 8 \
      --test_batch_size 8 \
      --n_epochs 100 \
      --seed ${SEED} \
      --use_quality_weight true \
      --degradation_mode multi \
      --deg_uncertainty_weight 0.001" \
      "${LOG_FILE}"

    echo ""
    echo "Finished ${MRTAG}, seed=${SEED}, log saved to ${LOG_FILE}"
  done
done

echo ""
echo "============================================================"
echo "All DARE-HME u=0.001 MOSI mr=0.2 / 0.3 runs finished."
echo "============================================================"
