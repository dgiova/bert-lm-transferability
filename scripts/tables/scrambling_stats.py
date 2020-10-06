import argparse
import glob
import os
import re
import itertools
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr, wilcoxon
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

colors = ['slateblue', 'darkorange', 'forestgreen', 'r']
# colors_ln = ['royalblue4', 'darkorange', 'forestgreen', 'r']
metrics = {'SST-2': 'Accuracy', 'QNLI': 'Accuracy', 'CoLA': 'MCC'}
x_labels = {'zapping': 'Number of preserved layers',
            'probing': 'Number of frozen layers'}
n_layers = {'start-layer': 13, 'only-layer': 12, 'block-start': 10}
plot_titles = {'start-layer': 'Partial reinitialization', 'block-start': 'Block reinitialization',
               'only-layer': 'Individual layer reinitialization'}
legend_labels = {'SST-2' : ['67k', '50k', '5k', '500'],
                 'QNLI' : ['105k', '50k', '5k', '500'],
                 'CoLA' : ['8.5k', '5k', '500']}

# Matlotlib settings
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['axes.titlesize'] = 'large'
mpl.rcParams['errorbar.capsize'] = 2

# DEBUGGING
def count_dirs(path, ln=False, experiment_name=''):
    return len(list(filter(lambda x: experiment_name in x, os.listdir(path))))

def split_reinit_ln(dirs):
    output_dirs = list(filter(lambda x: 'reinit-ln' not in x, dirs))
    ln_dirs = list(filter(lambda x: 'reinit-ln' in x, dirs))
    return output_dirs, ln_dirs

def get_scores(idxs, dirs):
    """returns list of scores given directories in seed folder"""
    formatted = list(zip(idxs, dirs))
    scores = []
    for l, name in sorted(formatted, key=lambda x: x[0]):
        with open(name + '/eval_results.txt', 'r') as f:
            score = float(f.readline().strip().split(' ')[-1])
            scores.append(score)
    return scores

def get_plt_args(x_axis, results, errorbar=False):
    """helper function to aggregate over trials"""
    means = [np.mean(x) for x in results]
    sds = [np.std(x) for x in results]
    if errorbar:
        return dict(x=x_axis, y=means, yerr=sds)
    else:
        return x_axis, means

def get_baseline(args, task_name, size):
    path = args.data_dir+'zapping'+'/'+task_name+'/'+size+'/*/start-layer-12/eval_results.txt'
    top = []
    for fn in glob.glob(path):
        with open(fn, 'r') as f:
            score = float(f.readline().strip().split(' ')[-1])
            top.append(score)
    path = args.data_dir+'zapping'+'/'+task_name+'/'+size+'/*/start-layer-0/eval_results.txt'
    bottom = []
    for fn in glob.glob(path):
        with open(fn, 'r') as f:
            score = float(f.readline().strip().split(' ')[-1])
            bottom.append(score)
    return np.mean(top), np.mean(bottom), np.std(top), np.std(bottom)



def calc_scrambling(tasks, args):
    d = defaultdict(list)
    task_scores = dict()
    # get top and bottom scores
    for i, task_name in enumerate(tasks):
        scores = []
        for seed in range(1, 11):
            res_file = args.data_dir+'scrambling'+'/'+task_name+'*/'+str(seed)+'/eval_results.txt'
            for fn in glob.glob(res_file):
                with open(fn, 'r') as f:
                    score = float(f.readline().strip().split(' ')[-1])
                    scores.append(score)
        task_scores[task_name] = scores
    # correlation test for all pairs
    for task1, task2 in itertools.combinations(tasks, 2):
        if task1!=task2:
            x1, x2 = task_scores[task1], task_scores[task2]
            sp, sp_pval = spearmanr(x1,x2)
            pr, pr_pval = pearsonr(x1,x2)
            wx, wx_pval = wilcoxon(x1,x2)
            d['Tasks compared'].append(task1+', '+task2)
            d["Spearman's correlation coefficient"].append('{:.2f}'.format(sp)+' ('+'{:.2f}'.format(sp_pval)+')')
            d["Pearson's correlation coefficient"].append('{:.2f}'.format(pr)+' ('+'{:.2f}'.format(pr_pval)+')')
    df = pd.DataFrame(d)
    print(df)
    print(df.to_latex(index=False))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--subset_size",
                        default="full",
                        type=str)
    parser.add_argument("--method_used",
                        default="zapping",
                        type=str,
                        help="zapping, scrambling, probing")
    parser.add_argument("--experiment_name",
                        default='start-layer',
                        type=str,
                        help="name of the experiment within the size/seed folder")
    parser.add_argument("--ln",
                        default=False,
                        type=bool,
                        help="True if you want to include layer norm experiment")
    parser.add_argument("--n_trials",
                        default=3,
                        type=int,
                        help="Number of trials run")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where the results are stored")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the plots are stored")
    parser.add_argument("--do_scrambling",
                        action="store_true",
                        help="Plot scrambling results")
    parser.add_argument("--plot_vertical",
                         action="store_true",
                         help="Plot results vertically")
    args = parser.parse_args()

    tasks = args.task_name.split(",")
    sizes = args.subset_size.split(",")

    calc_scrambling(tasks, args)

if __name__ == "__main__":
    main()
