#!/bin/bash

# . Scrambling
python3 scrambling_stats.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --data_dir 'OUTPUT_DIR  # please fill this in with the path to the output directory' \
  --method_used 'scrambling' \
  --output_dir '../../output/scrambling/' \
  --do_scrambling
