import argparse
import glob
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

colors = ['slateblue', 'darkorange', 'forestgreen', 'r']
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

# helper functions
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


# FIGURE 3
def plot_scrambling(tasks, args):
    fig, axs = plt.subplots(ncols=len(tasks), sharey=False, figsize=(8.5,5))
    task_scores = dict()
    task_baselines = dict()
    # get top and bottom scores
    for i, task_name in enumerate(tasks):
        scores = []
        for seed in range(1, 11):
            res_file = args.data_dir+args.method_used+'/'+task_name+'*/'+str(seed)+'/eval_results.txt'
            for fn in glob.glob(res_file):
                with open(fn, 'r') as f:
                    score = float(f.readline().strip().split(' ')[-1])
                    scores.append(score)
        task_scores[task_name] = scores
        # obtaining full and scratch models
        full_score, scratch_score, full_sd, scratch_sd = get_baseline(args, task_name, '5000')
        task_baselines[task_name] = [scratch_score, full_score]
        # formatting individual violin
        parts = axs[i].violinplot(scores, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('face')
            pc.set_alpha(0.25)
        axs[i].vlines(np.arange(1,2,1), min(scores), max(scores), color=colors[i], linestyle='-', lw=1.2)
        axs[i].set_ylabel(metrics[task_name], fontsize=12)
        axs[i].set_xticks(np.arange(1,2,1))
        xticklabel = task_name + ' ('+ metrics[task_name] + ')'
        axs[i].set_xticklabels([xticklabel], fontsize=12)
        axs[i].set_xlim(0.6, 1.4)
        # adding points for permutations
        axs[i].scatter([1]*10, scores, marker='o', color=colors[i], facecolors='none', 
                       s=40, label = 'permutations')
        # add full and scratch lines
        z = 2.667/np.sqrt(10) # 95% CI z score
        axs[i].axhspan(full_score-z*full_sd, full_score+z*full_sd, 
                       color='grey', label='_nolegend_', alpha=.25)
        axs[i].axhline(y=full_score, color=colors[i], label='full FT', ls='dashdot')
        axs[i].axhspan(scratch_score-z*scratch_sd, scratch_score+z*scratch_sd, 
                       color='grey', label='_nolegend_', alpha=.25)
        axs[i].axhline(y=scratch_score, color=colors[i], label='scratch FT', ls='dashed')
        # adding mean performance
        axs[i].axhline(y=np.mean(scores), xmin=.25, xmax=.75, color=colors[i], label='mean performance', ls='solid')
        # adding legend
        axs[i].legend(loc = 'upper right')
    # getting y axis limits
    acc_min = min(task_scores[tasks[0]] + task_scores[tasks[1]] + 
                  task_baselines[tasks[0]] + task_baselines[tasks[1]])
    acc_max = max(task_scores[tasks[0]] + task_scores[tasks[1]] +
                  task_baselines[tasks[0]] + task_baselines[tasks[1]])
    mcc_min = min(task_scores[tasks[2]] + task_baselines[tasks[2]])
    mcc_max = max(task_scores[tasks[2]] + task_baselines[tasks[2]]) 
    # space above and below plot
    m = .02
    # num ticks
    n_ticks = 10
    acc_yticks = np.linspace(acc_min,acc_max,n_ticks)
    mcc_yticks = np.linspace(mcc_min,mcc_max,n_ticks)
    # formatting for SST
    axs[0].set_ylim(acc_min-m,acc_max+m+0.035)
    axs[0].set_yticks(acc_yticks)
    axs[0].set_yticklabels(['%.2f'%round(i,2) for i in acc_yticks])
    # formatting for QNLI
    axs[1].set_ylim(acc_min-m,acc_max+m)
    axs[1].set_ylabel('')
    axs[1].set_yticks([])
    # formatting for CoLA
    axs[2].set_ylim(mcc_min-.07,mcc_max+.03+0.09)
    axs[2].set_yticks(mcc_yticks)
    axs[2].set_yticklabels(['%.2f'%round(i,2) for i in mcc_yticks])
    axs[2].yaxis.set_ticks_position('right')
    axs[2].yaxis.set_label_position('right')
    # formatting global plot
    fig.suptitle('Layer permutation', fontsize=16)
    plt.subplots_adjust(top=0.9, wspace=0.1)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(args.output_dir+'_test_plot.png', bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method_used",
                        default="zapping",
                        type=str,
                        help="zapping, scrambling, probing")
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
    args = parser.parse_args()
    task_name = 'SST-2,QNLI,CoLA'
    tasks = task_name.split(",")

    if args.do_scrambling:
        plot_scrambling(tasks, args)

if __name__ == "__main__":
    main()
