#!/bin/bash

export GLUE_DIR=DATA_DIR  # please fill this in with the path to the glue data directory
OUTPUT_DIR=OUTPUT_DIR  # please fill this in with the path to the output directory
CUDA_DEVICE=CUDA_DEVICE # please set cuda device

N_TRIALS=10
BLOCK_SIZE=3
LOCALIZED_END=$((11-$BLOCK_SIZE+1))



for TASK_NAME in 'SST-2' 'QNLI' 'CoLA'
do
  for SIZE in 5000
  do
    for SEED in $(seq 1 $N_TRIALS)
    do
      for START in $(seq 0 $LOCALIZED_END)
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
          --output_dir $OUTPUT_DIR/zapping/$TASK_NAME/$SIZE/$SEED/block-$BLOCK_SIZE-start-$START/ \
          --overwrite_output_dir \
          --save_steps 2000 \
          --start_layer $START \
          --end_layer $(($START + $BLOCK_SIZE - 1)) \
          --seed $SEED \
          --subset_size $SIZE
      done
    done
  done
done

# Inverse block
for TASK_NAME in 'SST-2' 'QNLI' 'CoLA'
do
  for SIZE in 5000
  do
    for SEED in $(seq 1 $N_TRIALS)
    do
      for START in $(seq 0 $LOCALIZED_END)
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
          --output_dir $OUTPUT_DIR/zapping/$TASK_NAME/$SIZE/$SEED/block-$BLOCK_SIZE-inv-start-$START/ \
          --overwrite_output_dir \
          --save_steps 2000 \
          --start_layer $START \
          --end_layer $(($START + $BLOCK_SIZE - 1)) \
          --seed $SEED \
          --subset_size $SIZE \
          --do_inv_local_reinit
      done
    done
  done
done


