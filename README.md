# Investigating Transferability in Pretrained Language Models
**Alex Tamkin, Trisha Singh, Davide Giovanardi, Noah Goodman** <br>
**Stanford University** <br>

How does language model pretraining help transfer learning? Code for experiments and figures.

## Scripts
Here, you can find the scripts to run experiments and produce visualizations.
When running an experiment for the first time, please make sure to edit the `DATA_DIR` and `OUTPUT_DIR` variables within the 
desired bash script (see below).

To get started, run `download_glue_data.sh` to get the GLUE datasets and be ready to run the experiments.

### Experiments
Here, you can find the bash scripts to replicate the experiments in the paper:
- `run_progressive.sh`: progressive reinitialization method.
- `run_block.sh`: localized reinitialization with fixed window size (block). The default block size is 3.
- `run_probing.sh`: probing method as described in the paper.
- `run_scrambling.sh`: finetuning the pretrained model after permuting the layers.
- `run_lr-5x.sh`: progressive reinitialization method with new learning rate.
- `run_keep-ln.sh`: progressive reinitialization method with the layer-norm parameters preserved during fine-tuning.

### Figures
- `figure.sh`: produces all the figures included in the paper. The plotting methods are stored in `make_plot.py` and `make_violin.py`

### Tables
- `scrambling_stats.sh`: performs a correlation analysis related to the layers permutation experiment.

## Examples
This folder contains the python files that take care of loading the BERT pretrained layers, initialize our experiments, and finetune the model. You won't need to modify anything here (unless you wish to try different approaches). These files are called by the wrapper bash scripts discussed above in **Experiments**.
- `run_glue.py`: progressive (regular, frozen layer-norm) and localized reinitialization, scrambling.
- `run_glue_lr.py`: progressive reinitialization with new learning rate.
- `run_glue_probing.py`: probing method.
