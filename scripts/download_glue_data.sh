#!/bin/bash

export DIR=DIR            # please fill this in with the path to the main working directory
export GLUE_DIR=DATA_DIR  # please fill this in with the path to the glue data directory

for TASK_NAME in 'CoLA' 'SST' 'QNLI'
do
  python3 $DIR/scripts/download_glue_data.py --data_dir $GLUE_DIR --tasks $TASK_NAME
done