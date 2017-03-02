import os
import sys
import importlib
import imp
import pdb
import argparse
import numpy             as np
import numpy.random      as npr
import numpy.linalg      as npla
import matplotlib        as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from spearmint.visualizations         import plots_2d
from spearmint.utils.parsing          import parse_config_file
from spearmint.utils.parsing          import parse_tasks_from_jobs
from spearmint.utils.parsing          import repeat_experiment_name
from spearmint.utils.parsing          import get_objectives_and_constraints
from spearmint.utils.parsing          import DEFAULT_TASK_NAME
from spearmint.utils.database.mongodb import MongoDB
from spearmint.tasks.input_space      import InputSpace
from spearmint.tasks.input_space      import paramify_no_types
from spearmint.main                   import load_jobs
import matplotlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', type=str, nargs='+')
    parser.add_argument('--opt', type=float)
    parser.add_argument('--niter', type=int)
    parser.add_argument('--nruns', type=int, default=1)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--xlabel', type=str)
    parser.add_argument('--ylabel', type=str)
    parser.add_argument('--legend',  type=str, nargs='*')
    parser.add_argument('--title', type=str)
    parser.add_argument('--plotname', type=str)



    args = parser.parse_args()

    # let's unpack some of these argumnents
    directory_list = args.expdir
    num_iters = args.niter
    num_runs = args.nruns
    title = args.title
    optimum_val = args.opt

    if args.xlabel is None:
        xlabel = "Number of Samples"
    else:
        xlabel = args.xlabel

    if args.ylabel is None:
        ylabel = "log Immediate Regret"
    else:
        ylabel = args.ylabel

    if args.savedir is None:
        savedir = directory_list[0]
    else:
        savedir = args.savedir

    if args.plotname is None:
        plot_name = 'regret.pdf'
    else:
        plot_name = args.plotname
    if args.legend is None:
        legend_on = False
        legend_labels = len(directory_list)*['']
    else:
        legend_on =True
        # legend_labels = args.legend
        legend_labels = ['TS', 'EI', 'parallel TS', 'parallel EI']

    # fontsize
    font = {'size': 18}
    matplotlib.rc('font', **font)

    # set up the colormap
    num_plots = len(directory_list)
    colors = [mpl.cm.jet(_) for _ in np.linspace(0.0, 1.0, num_plots)]

    fig = plt.figure(1)
    plt.clf()

    for expt_dir, c, legend_label in zip(directory_list, range(num_plots), legend_labels):
        regret = np.empty(shape=(num_runs, num_iters))
        for i in xrange(num_runs):
            options         = parse_config_file(expt_dir, 'config.json')
            experiment_name = options["experiment-name"]
            experiment_name = repeat_experiment_name(experiment_name, i)
            # num_iters = int(options["max_iterations"])

            input_space     = InputSpace(options["variables"])
            chooser_module  = importlib.import_module('spearmint.choosers.' + options['chooser'])
            chooser         = chooser_module.init(input_space, options)
            db              = MongoDB(database_address=options['database']['address'])
            jobs            = load_jobs(db, experiment_name)
            hypers          = db.load(experiment_name, 'hypers')
            tasks           = parse_tasks_from_jobs(jobs, experiment_name, options, input_space)



            for task_name, task in tasks.iteritems():
                temp_min = np.inf
                for j in xrange(num_iters):
                    # print task.values
                    # print i, j
                    temp_min = np.min([temp_min, task.values[j]])
                    regret[i, j] = temp_min

        regret = np.log10(np.abs(regret - optimum_val))
        mean_regret = np.mean(regret, 0)
        std_regret = np.std(regret, 0)/np.sqrt(num_runs)


        plt.plot(range(1,len(mean_regret)+1), mean_regret, color=colors[c], lw=2, zorder=9, label=legend_label)
        plt.fill_between(range(1,len(mean_regret)+1), mean_regret - std_regret, mean_regret + std_regret, alpha=0.4, color=colors[c])

    if legend_on:
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


    folder = os.path.join(savedir, 'plots')
    if not os.path.isdir(folder):
        os.makedirs(folder)
    save_name = os.path.join(folder, plot_name)
    plt.savefig(save_name)


if __name__ == '__main__':
    # main(*sys.argv[1:])
    main()