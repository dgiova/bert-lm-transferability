#!/bin/bash

# FIGURE 1
# Progressive reinitialization and probing 
python3 make_plot.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '50000','5000','500','probing' \
  --data_dir 'OUTPUT_DIR  # please fill this in with the path to the output directory' \
  --output_dir '../../figures/result/' \
  --plot_vertical

# FIGURE 2
# Localized reinitialization (with blocks reinitialized and preserved)
python3 make_plot.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000' \
  --data_dir 'OUTPUT_DIR  # please fill this in with the path to the output directory' \
  --output_dir '../../figures/result/' \
  --experiment_name 'block-start-*-reinit-ln','inv-local-start-*'\
  --plot_block

# FIGURE 3
# Changing order of pretrained layers
python3 make_violin.py \
\  --data_dir 'OUTPUT_DIR  # please fill this in with the path to the output directory' \
  --method_used 'scrambling' \
  --output_dir '../../figures/result/' \
  --do_scrambling

# FIGURE 4
# Scatter plot of partial reinitialization for 500 examples
python3 make_plot.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '500' \
  --data_dir 'OUTPUT_DIR  # please fill this in with the path to the output directory' \
  --output_dir '../../figures/result/' \
  --plot_scatter

# FIGURE 5
# Localized reinitialization with individual layers
python3 make_plot.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000' \
  --data_dir 'OUTPUT_DIR  # please fill this in with the path to the output directory' \
  --output_dir '../../figures/result/' \
  --experiment_name 'only-layer-*-reinit-ln' \
  --plot_individual

# FIGURE 6
# Progressive reinitialization with 5x larger learning rate
python3 make_plot.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000-lr-5x','5000-selective-layer_norm' \
  --data_dir 'OUTPUT_DIR  # please fill this in with the path to the output directory' \
  --output_dir '../../figures/result/' \
  --lr_5x

# FIGURE 7
# Progressive reinitialization with layer norm parameters kept
python3 make_plot.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000','5000-selective-layer_norm' \
  --data_dir 'OUTPUT_DIR  # please fill this in with the path to the output directory' \
  --output_dir '../../figures/result/' \
  --keep_ln

