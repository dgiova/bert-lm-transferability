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
plot_titles = {'start-layer': 'Progressive reinitialization', 'block-start': 'Block reinitialization',
               'only-layer': 'Individual layer reinitialization'}
legend_labels = {'SST-2' : ['50k', '5k', '500','probing'],
                 'QNLI' : ['50k', '5k', '500','probing'],
                 'CoLA' : ['50k', '5k', '500','probing']}
ln_legend_labels = ['layer norm kept', 'layer norm reinit']
lr_legend_labels = ['5x learning rate', 'default learning rate']

# Matlotlib settings
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['axes.titlesize'] = 'large'
mpl.rcParams['errorbar.capsize'] = 2
smallest_font = 16
alpha_denom = 0.75

# Helper functions
def get_scores(idxs, dirs):
    """returns list of scores given directories in seed folder"""
    formatted = list(zip(idxs, dirs))
    scores = []
    for l, name in sorted(formatted, key=lambda x: x[0]):
        with open(name + '/eval_results.txt', 'r') as f:
            score = float(f.readline().strip().split(' ')[-1])
            scores.append(score)
    return scores


def get_baseline(args, task_name, size):
    """gets scores for embedding only and fully finetuned model"""
    path = 'OUTPUT_DIR  # please fill this in with the path to the output directory/'+args.method_used+'/'
         + task_name+'/'+'5000-selective-layer_norm'+'/*/start-layer-12/eval_results.txt'
    top = []
    for fn in glob.glob(path):
        with open(fn, 'r') as f:
            score = float(f.readline().strip().split(' ')[-1])
            top.append(score)
    path = 'OUTPUT_DIR  # please fill this in with the path to the output directory/'+args.method_used+'/'
         + task_name+'/'+'5000-selective-layer_norm'+'/*/start-layer-0/eval_results.txt'
    bottom = []
    for fn in glob.glob(path):
        with open(fn, 'r') as f:
            score = float(f.readline().strip().split(' ')[-1])
            bottom.append(score)
    return np.mean(top), np.mean(bottom)


def get_plt_args(x_axis, results, n_trials=3, t_val=2.667, errorbar=False):
    """aggregate over trials, return mean and standard deviation"""
    means = [np.mean(x) for x in results]
    sds = [(t_val / np.sqrt(n_trials)) * np.std(x) for x in results]
    if errorbar:
        return dict(x=x_axis, y=means, yerr=sds)
    else:
        return x_axis, means


# Plotting functions

# FIGURE 1
ls_progressive = {0: 'solid', 1: 'dotted', 2:'dashdot', 3:'dashed'}
def plot_vertical_with_probing(tasks, sizes, args):
    fig, axs = plt.subplots(ncols=len(tasks), sharey=False, figsize=(9,6.5))
    for i, task_name in enumerate(tasks):
        for s, size in enumerate(sizes):
            final_scores = []
            if size == '500':
                n_trials = 50
            elif size.startswith('probing'):
                n_trials = 10
            else:
                n_trials = args.n_trials
            for seed in range(1, n_trials+1):
                # output dirs contains list of layers without layer norm
                if size.startswith('probing'):
                    path = args.data_dir+args.method_used+'/'+task_name+'/'+size+'/'+str(seed)+'/end-layer-*'
                elif size == '5000':
                    path = args.data_dir+args.method_used+'/'+task_name+'/5000-selective-laye*/'+str(seed)+'/'+args.experiment_name+'-*'
                else:
                    path = args.data_dir+args.method_used+'/'+task_name+'/'+size+'/'+str(seed)+'/'+args.experiment_name+'-*'
                output_dirs = glob.glob(path)
                idxs = [int(l.split('-')[-1]) for l in output_dirs]  # get the start layer
                scores = get_scores(idxs, output_dirs)
                final_scores.append(scores)
            results = list(zip(*final_scores))
            if len(idxs) < 1:
                continue
            if size.startswith('probing'):
                x_axis = [min(idxs) + s - 1 for s in sorted(idxs, reverse=False)]
            else:
                x_axis_ = [min(idxs) + s for s in sorted(idxs, reverse=False)]
                x_axis = [min(idxs) + s for s in sorted(idxs, reverse=False)]
            plt_args = get_plt_args(x_axis, results, n_trials, errorbar=True)
            if size == 'full':
                axs[i].errorbar(**plt_args, marker=',', label=legend_labels[task_name][s],
                                markerfacecolor=colors[i], markersize=4, #fillstyle='none',
                                color=colors[i], linewidth=1, linestyle='dashed')
            elif size.startswith('probing'):
                eb = axs[i].errorbar(**plt_args, marker=',', label=legend_labels[task_name][s],
                                markerfacecolor=colors[i], markersize=4, #fillstyle='none',
                                     color=colors[i], linewidth=1, linestyle='dashed', alpha=1/2)
                eb[-1][0].set_linestyle('--')
            else:
                axs[i].errorbar(**plt_args, marker='o', label=legend_labels[task_name][s],
                                markerfacecolor=colors[i], markersize=4, #linestyle=ls_progressive[s],#fillstyle='none',
                                color=colors[i], linewidth=1+(1-.25*s), alpha=min(1,1/(s/2+alpha_denom)))
        axs[i].legend(loc='upper left')
        axs[i].set_title(task_name+' ('+metrics[task_name]+')')
        axs[i].set_xticks(range(min(x_axis_), max(x_axis_) + 1, 4))
    axs[0].set_ylim(0.45,1.00)
    axs[1].set_ylim(0.45,1.00)
    axs[0].set_ylabel(metrics['SST-2'], fontsize=smallest_font)
    axs[2].set_ylabel(metrics['CoLA'], fontsize=smallest_font)
    axs[2].yaxis.set_label_position("right")
    axs[1].set_xlabel(x_labels[args.method_used], fontsize=smallest_font+2)
    fig.suptitle(plot_titles[args.experiment_name], fontsize=smallest_font+4)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(args.output_dir+'progressive_with_probing.png', dpi=300, bbox_inches='tight')


# FIGURE 2
legend_block = {0: 'block reinitialized', 1: 'block preserved'}
ls_block = {0: 'solid', 1: 'dotted'}
def plot_block(tasks, sizes, experiments, args):
    fig, axs = plt.subplots(nrows=len(tasks), sharex=True, figsize=(9,6.5))
    for i, task_name in enumerate(tasks):
        for s, size in enumerate(sizes):
            top, bottom = get_baseline(args, task_name, size)
            axs[i].plot(range(10), [top]*10, color=colors[i], label='full FT', ls='dashdot')
            axs[i].plot(range(10), [bottom]*10, color=colors[i], label='emb only FT', ls='dashed')
            for e, experiment in enumerate(experiments):
                if experiment == 'block-start-*-reinit-ln':
                    n_trials = 10
                else:
                    n_trials = args.n_trials
                final_scores = []
                for seed in range(1, n_trials+1):
                    output_dirs = glob.glob(args.data_dir+args.method_used+'/'+task_name+'/'+
                                            size+'/'+str(seed)+'/'+experiment)
                    if experiment[-1] == 'n':
                        idxs = [int(l.split('-')[-3]) for l in output_dirs]  # get the start layer
                    else:
                        idxs = [int(l.split('-')[-1]) for l in output_dirs]  # get the start layer
                    scores = get_scores(idxs, output_dirs)
                    final_scores.append(scores)
                results = list(zip(*final_scores))
                if len(idxs) < 1:
                    continue
                x_axis = [min(idxs) + s for s in sorted(idxs, reverse=False)]
                plt_args = get_plt_args(x_axis, results, errorbar=True, n_trials=n_trials)
                axs[i].errorbar(**plt_args, marker='o',  label=legend_block[e],
                                markerfacecolor=colors[i], markersize=4,
                                color=colors[i], linewidth=1, linestyle=ls_block[e])
        # reorder legend items
        handles, labels = axs[i].get_legend_handles_labels()
        order = [0,2,3,1]
        axs[i].legend(handles=[handles[j] for j in order], labels=[labels[j] for j in order], loc='center right')
        axs[i].set_title(task_name) 
        axs[i].set_ylabel(metrics[task_name], fontsize=smallest_font)
    axs[i].set_xticks(range(min(x_axis), max(x_axis) + 1))
    x_tickslabels = [str(i) + '-' + str(i + 1) + '-' + str(i + 2) for i in range(10)]
    axs[i].set_xticklabels(x_tickslabels, rotation=30)
    axs[2].set_xlabel('Layers', fontsize=smallest_font+2)
    fig.suptitle('Localized reinitialization', fontsize=smallest_font+4)
    fig.subplots_adjust(hspace=0.25)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(args.output_dir+'localized_may25.png', dpi=300, bbox_inches='tight')


# FIGURE 4
def plot_scatter(tasks, sizes, args):
    fig, axs = plt.subplots(ncols=len(tasks), sharey=False, figsize=(9,6.5))
    for alpha in [0.15]:
        for i, task_name in enumerate(tasks):
            for s, size in enumerate(sizes):
                final_scores = []
                if size == '500':
                    n_trials = 50
                else:
                    n_trials = args.n_trials
                for seed in range(1, n_trials+1):
                    # output dirs contains list of layers without layer norm
                    path = args.data_dir+args.method_used+'/'+task_name+'/'+size+'/'+str(seed)+'/'+args.experiment_name+'-*'
                    output_dirs = glob.glob(path)
                    idxs = [int(l.split('-')[-1]) for l in output_dirs]  # get the start layer
                    scores = get_scores(idxs, output_dirs)
                    final_scores.append(scores)
                results = list(zip(*final_scores))
                if len(idxs) < 1:
                    continue
                x_axis = [min(idxs) + s for s in sorted(idxs, reverse=False)]
                for xe, ye in zip(x_axis, results):
                    axs[i].scatter([xe]*len(ye), ye, c=colors[i], alpha=alpha)
                x_line, y_line = get_plt_args(x_axis, results)
                axs[i].plot(x_line, y_line, marker='.', 
                                markerfacecolor=colors[i], markersize=4, 
                                color=colors[i], linewidth=1, linestyle='dashed', alpha=0.75)
            axs[i].set_title(task_name+' ('+metrics[task_name]+')')
            axs[i].set_xticks(range(min(x_axis), max(x_axis) + 1, 4))
        axs[0].set_ylim(0.45,0.90)
        axs[1].set_ylim(0.45,0.90)
        axs[0].set_ylabel(metrics['SST-2'], fontsize=smallest_font)
        axs[2].set_ylabel(metrics['CoLA'], fontsize=smallest_font)
        axs[2].yaxis.set_label_position("right")
        axs[1].set_xlabel(x_labels[args.method_used], fontsize=smallest_font+2)
        fig.suptitle('Progressive reinitialization (500 samples)', fontsize=smallest_font+4)
        # more space between subplots
        fig.subplots_adjust(wspace=0.25)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        plt.savefig(args.output_dir+str(alpha)+'_scatter.png', dpi=300, bbox_inches='tight')


# FIGURE 5
legend_individual = {0: 'layer reinitialized', 1: 'block preserved'}
ls_individual = {0: 'solid', 1: 'dotted'}
def plot_individual(tasks, sizes, experiments, args):
    fig, axs = plt.subplots(nrows=len(tasks), sharex=True, figsize=(9,6.5))
    for i, task_name in enumerate(tasks):
        for s, size in enumerate(sizes):
            for e, experiment in enumerate(experiments):
                final_scores = []
                for seed in range(1, args.n_trials+1):
                    # output dirs contains list of layers without layer norm
                    output_dirs = glob.glob(args.data_dir+args.method_used+'/'+task_name+'/'+
                                            size+'/'+str(seed)+'/'+experiment)
                    if experiment[-1] == 'n':
                        idxs = [int(l.split('-')[-3]) for l in output_dirs]  # get the start layer
                    else:
                        idxs = [int(l.split('-')[-1]) for l in output_dirs]  # get the start layer
                    scores = get_scores(idxs, output_dirs)
                    final_scores.append(scores)
                results = list(zip(*final_scores))
                if len(idxs) < 1:
                    continue
                x_axis = [min(idxs) + s for s in sorted(idxs, reverse=False)]
                plt_args = get_plt_args(x_axis, results, errorbar=True)
                axs[i].errorbar(**plt_args, marker='o',  label=legend_individual[e],
                                markerfacecolor=colors[i], markersize=4, 
                                color=colors[i], linewidth=1, linestyle=ls_individual[e])
            top, bottom = get_baseline(args, task_name, size)
            axs[i].plot(range(12), [top]*12, color=colors[i], label='full FT', ls='dashdot')
            axs[i].plot(range(12), [bottom]*12, color=colors[i], label='emb only FT', ls='dashed')
        handles, labels = axs[i].get_legend_handles_labels()
        order = [0,2,1]
        axs[i].legend(handles=[handles[j] for j in order], labels=[labels[j] for j in order], loc='center right')
        axs[i].set_title(task_name) 
        axs[i].set_ylabel(metrics[task_name], fontsize=smallest_font)
    axs[i].set_xticks(range(12))
    axs[i].set_xticklabels(range(1,13))
    axs[2].set_xlabel('Layers', fontsize=smallest_font+2)
    fig.suptitle('Individual layer reinitialization', fontsize=smallest_font+4)
    fig.subplots_adjust(hspace=0.25)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(args.output_dir+'individual.png', dpi=300, bbox_inches='tight')


# FIGURE 6
ls_5x = {0: 'solid', 1: 'dotted'}
def plot_lr(tasks, sizes, args):
    fig, axs = plt.subplots(ncols=len(tasks), sharey=False, figsize=(9,6.5))
    for i, task_name in enumerate(tasks):
        for s, size in enumerate(sizes):
            final_scores = []
            n_trials = args.n_trials
            for seed in range(1, n_trials+1):
                path = args.data_dir+args.method_used+'/'+task_name+'/'+size+'/'+str(seed)+'/'+args.experiment_name+'-*'
                output_dirs = glob.glob(path)
                idxs = [int(l.split('-')[-1]) for l in output_dirs]  # get the start layer
                scores = get_scores(idxs, output_dirs)
                final_scores.append(scores)
            results = list(zip(*final_scores))
            if len(idxs) < 1:
                continue
            x_axis_ = [min(idxs) + s for s in sorted(idxs, reverse=False)]
            plt_args = get_plt_args(x_axis_, results, n_trials, errorbar=True)
            axs[i].errorbar(**plt_args, marker='o', 
                            label=lr_legend_labels[s],
                            markerfacecolor=colors[i], markersize=4, linestyle = ls_5x[s],
                            color=colors[i], linewidth=1, alpha=min(1,1/(s/2+alpha_denom)))
        axs[i].legend(loc='upper left')
        axs[i].set_title(task_name+' ('+metrics[task_name]+')')
        axs[i].set_xticks(range(min(x_axis_), max(x_axis_) + 1, 4))
    axs[0].set_ylim(0.35,1.00)
    axs[1].set_ylim(0.35,1.00)
    axs[2].yaxis.set_label_position("right")
    axs[1].set_xlabel(x_labels[args.method_used], fontsize=smallest_font+2)
    fig.suptitle(plot_titles[args.experiment_name], fontsize=smallest_font+4)
    fig.subplots_adjust(wspace=0.25)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(args.output_dir+'progressive_lr_5x.png', dpi=300, bbox_inches='tight')


# FIGURE 7
ls_ln = {0: 'solid', 1: 'dotted'}
def plot_keep_ln(tasks, sizes, args):
    fig, axs = plt.subplots(ncols=len(tasks), sharey=False, figsize=(9,6.5))
    for i, task_name in enumerate(tasks):
        for s, size in enumerate(sizes):
            final_scores = []
            n_trials = args.n_trials
            for seed in range(1, n_trials+1):
                path = args.data_dir+args.method_used+'/'+task_name+'/'+size+'/'+str(seed)+'/'+args.experiment_name+'-*'
                output_dirs = glob.glob(path)
                idxs = [int(l.split('-')[-1]) for l in output_dirs]  # get the start layer
                scores = get_scores(idxs, output_dirs)
                final_scores.append(scores)
            results = list(zip(*final_scores))
            if len(idxs) < 1:
                continue
            x_axis_ = [min(idxs) + s for s in sorted(idxs, reverse=False)]
            plt_args = get_plt_args(x_axis_, results, n_trials, errorbar=True)
            axs[i].errorbar(**plt_args, marker='o', 
                            label=ln_legend_labels[s],
                            markerfacecolor=colors[i], markersize=4, linestyle = ls_ln[s], 
                            color=colors[i], linewidth=1, alpha=min(1,1/(s/2+alpha_denom)))
        axs[i].legend(loc='upper left')
        axs[i].set_title(task_name+' ('+metrics[task_name]+')')
        axs[i].set_xticks(range(min(x_axis_), max(x_axis_) + 1, 4))
    axs[0].set_ylim(0.55,0.95)
    axs[1].set_ylim(0.55,0.95)
    axs[2].yaxis.set_label_position("right")
    axs[1].set_xlabel(x_labels[args.method_used], fontsize=smallest_font+2)
    fig.suptitle(plot_titles[args.experiment_name], fontsize=smallest_font+4)
    fig.subplots_adjust(wspace=0.25)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(args.output_dir+'progressive_keep_ln.png', dpi=300, bbox_inches='tight')


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
    parser.add_argument("--plot_block",
                         action="store_true",
                         help="Plot localized results")
    parser.add_argument("--plot_probing",
                         action="store_true",
                         help="Plot probing results")
    parser.add_argument("--plot_scatter",
                         action="store_true",
                         help="Plot scatter results")    
    parser.add_argument("--keep_ln",
                         action="store_true",
                         help="Plot results when ln params not reinitialized")    
    parser.add_argument("--plot_individual",
                         action="store_true",
                         help="Plot individual layer reinit results")    
    parser.add_argument("--lr_5x",
                         action="store_true",
                         help="Plot results when learning rate 5x")    
    args = parser.parse_args()

    tasks = args.task_name.split(",")
    sizes = args.subset_size.split(",")

    # Figure 1
    if args.plot_vertical:
        plot_vertical_with_probing(tasks, sizes, args)
    # Figure 2
    elif args.plot_block:
        experiments = args.experiment_name.split(',')
        plot_block(tasks, sizes, experiments, args)
    # Figure 4
    elif args.plot_scatter:
        plot_scatter(tasks, sizes, args)
    # Figure 5
    elif args.plot_individual:
        experiments = args.experiment_name.split(',')
        plot_individual(tasks, sizes, experiments, args)
    # Figure 6
    elif args.lr_5x:
        plot_lr(tasks, sizes, args)
    # Figure 7
    elif args.keep_ln:
        plot_keep_ln(tasks, sizes, args)

if __name__ == "__main__":
    main()
