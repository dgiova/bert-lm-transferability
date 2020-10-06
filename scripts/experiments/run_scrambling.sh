#!/bin/bash

export GLUE_DIR=DATA_DIR  # please fill this in with the path to the glue data directory
OUTPUT_DIR=OUTPUT_DIR  # please fill this in with the path to the output directory
CUDA_DEVICE=CUDA_DEVICE # please set cuda device

SCRAMBLING_TRIALS=10

for TASK_NAME in 'SST-2' 'CoLA' 'QNLI'
do
  for PERM_SEED in $(seq 1 $SCRAMBLING_TRIALS)
  do
      CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 ../../examples/run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $OUTPUT_DIR/scrambling/$TASK_NAME/$PERM_SEED/ \
        --overwrite_output_dir \
        --permutation_seed $PERM_SEED \
        --save_steps 2000 \
        --subset_size 5000 \
        --seed 1
  done
done
