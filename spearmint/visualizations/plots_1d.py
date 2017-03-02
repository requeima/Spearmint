import os
import sys
import importlib
import numpy             as np
import matplotlib        as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import glob
import re

from collections                      import defaultdict
from spearmint.utils.parsing          import parse_config_file
from spearmint.utils.parsing          import parse_tasks_from_jobs
from spearmint.utils.database.mongodb import MongoDB
from spearmint.tasks.input_space      import InputSpace
from spearmint.main                   import load_jobs
from spearmint.main                   import print_hypers
from spearmint.models.abstract_model  import function_over_hypers, function_over_hypers_thompson_sampling
from spearmint.utils.parsing          import repeat_experiment_name

cmap = mpl.cm.get_cmap('Blues')

x_size = 16
x_width = 4
o_data_size = 12
star_size = 30
star_width = 4
nGridPoints = 200

def incriment_filename(directory, file_name):
    full_name = os.path.join(directory, file_name)
    plot_list = glob.glob(full_name + '*')
    plot_number_list = []

    if not plot_list:
        save_name = full_name + '_0.pdf'
    else:
        for s in plot_list:
            plot_number_list.append(int(re.findall(r'\d+', os.path.abspath(s))[1]))
        plot_number = max(plot_number_list) + 1
        save_name = full_name + '_' + str(plot_number) + '.pdf'
    return save_name

def get_ready_to_plot(input_space, current_best):

    xmin = input_space.variables_meta.values()[0]['min']
    xmax = input_space.variables_meta.values()[0]['max']
    bounds = (xmin,xmax)
    xlabel = input_space.variables_meta.keys()[0]

    if current_best is not None:
        current_best = input_space.from_unit(current_best)
        current_best = current_best.flatten()

    x_grid = np.array([[_] for _ in np.linspace(0.0, 1.0, nGridPoints)])
    # y_grid = np.linspace(0.0, 1.0, nGridPoints)
    # X, Y = np.meshgrid(x_grid, y_grid)
    # flat_grid = np.hstack((X.flatten()[:,np.newaxis], Y.flatten()[:,np.newaxis]))

    def setaxes():
        plt.xlim((xmin,xmax))
        plt.xticks(np.linspace(xmin, xmax, 6), size=20)

        if xlabel is not None:
            plt.xlabel(xlabel, size=35)
        plt.tight_layout()

    mappedX = input_space.from_unit(x_grid).flatten()
    return xmin, xmax, x_grid, mappedX, current_best, bounds, setaxes


def plot_mean_and_var(model, directory, input_space, current_best=None):
    xmin, xmax, x_grid, mappedX, current_best, bounds, setaxes = get_ready_to_plot(input_space, current_best)

    # figpath = os.path.join(directory, 'posterior')

    ############### ---------------------------------------- ############
    ###############                                          ############
    ###############           PLOT GP FUNCTION               ############
    ###############                                          ############
    ############### ---------------------------------------- ############

    fig = plt.figure(1)
    plt.clf()

    if model.options['acquisition'] == 'TS':
        mean, var = model.function_over_hypers_thompson_sampling(model.predict, x_grid)
    else:
        mean, var = model.function_over_hypers(model.predict, x_grid)

    x_data = model.observed_inputs
    mapped_x_data = input_space.from_unit(x_data)
    y_data = model.observed_values

    # plot and save
    plt.plot(mappedX, mean, 'k', lw=2, zorder=9)
    plt.fill_between(mappedX, mean - np.sqrt(var), mean + np.sqrt(var), alpha=0.4, color='b')
    plt.scatter(mapped_x_data, y_data, color='k', s=50, zorder=10)
    setaxes()


    save_name = incriment_filename(directory, 'posterior_GP')
    plt.savefig(save_name)


    ############## ---------------------------------------- ############
    ##############                                          ############
    ##############   PLOT OBJECTIVE ACQUISITION FUNCTION    ############
    ##############                                          ############
    ############## ---------------------------------------- ############


def plot_acquisition_function(model, chooser, directory, input_space, current_best_location, current_best_value):
    # if there is non-competitive decoupling, then the tasks might have different acquisition funcitons
    # so we see if there are multiple acquisition functions in play here...
    # this is a bit hacky and needs to be handled better later
    # acq_funcs = list({chooser.acquisition_functions[task_name]["name"] for task_name in chooser.options["tasks"]})
    # tasks_to_plot = [chooser.tasks.keys()[0]] if len(acq_funcs) == 1 else chooser.tasks.keys()

    # for task_name in tasks_to_plot:

    suggestion = chooser.suggest(chooser.tasks.keys())[0]  # must do this before computing acq

    # acq_name = chooser.acquisition_functions[task_name]["name"]
    # acq_fun  = chooser.acquisition_functions[task_name]["class"](chooser.num_dims, DEBUG_input_space=input_space)
    # acq_fun = chooser.acq[task_name] # get the cached one, so that randomness like x* sampling in PES stays the same
    # above is the non-cached one, will sample a different x*
    acq_name = chooser.acquisition_function_name
    acq_fun = chooser.acq

    xmin, xmax, x_grid, mappedX, current_best, bounds, setaxes = get_ready_to_plot(input_space, current_best_location)

    # flat_grid, mappedX, mappedY, mapped_current_best, bounds, setaxes = get_ready_to_plot(input_space,
    #                                                                                       current_best_location)
    fig = plt.figure(2)
    plt.clf()

    if model.options['acquisition'] == 'TS':
        acq = function_over_hypers_thompson_sampling(chooser.models.values(), acq_fun.acquisition,
                                                     chooser.objective_model_dict, chooser.constraint_models_dict,
                                                     x_grid, current_best_value, compute_grad=False)
    else:
        acq = function_over_hypers(chooser.models.values(), acq_fun.acquisition,
                                   chooser.objective_model_dict, chooser.constraint_models_dict,
                                   x_grid, current_best_value, compute_grad=False)
    # best_acq_index = np.argmax(acq)
    if model.options['acquisition'] == 'TS':
        plt.plot(mappedX, -acq)
    else:
        plt.plot(mappedX, acq)
    # plot the suggestion
    # plt.axvline(x=input_space.from_unit(np.array([[suggestion]])))

    plt.axvline(x=suggestion)
    # best_acq_location = flat_grid[best_acq_index]
    # mapped_best_acq_location = input_space.from_unit(best_acq_location).flatten()
    # plt.plot(mapped_best_acq_location[0], mapped_best_acq_location[1], color='green', marker='x', markersize=10, markeredgewidth=0.5)
    # plt.plot(mapped_best_acq_location[0], mapped_best_acq_location[1], color='red', marker='*', markersize=star_size)#, markeredgecolor='orange') # markeredgewidth=star_width,

    # see the chooser's grid overlay
    # plt.plot(chooser.grid[:,0], chooser.grid[:,1], 'r.')

    setaxes()
    # dire = directory if len(acq_funcs) == 1 else os.path.join(directory, task_name)

    save_name = incriment_filename(directory, 'acquisition_function')
    plt.savefig(save_name)

    # dire = directory
    # file_name = os.path.join(dire, '%s_acquisition_function_0_.pdf' % acq_name)
    # if not os.path.isfile(file_name): # this is a hack and only works for 0-100
    #     plt.savefig(file_name)
    # else:
    #     file_name = file_name[]


def main(expt_dir, repeat=None):
    options = parse_config_file(expt_dir, 'config.json')
    experiment_name = options["experiment-name"]

    if repeat is not None:
        experiment_name = repeat_experiment_name(experiment_name,repeat)

    input_space = InputSpace(options["variables"])

    chooser_module = importlib.import_module('spearmint.choosers.' + options['chooser'])
    chooser = chooser_module.init(input_space, options)

    db = MongoDB(database_address=options['database']['address'])

    jobs = load_jobs(db, experiment_name)
    hypers = db.load(experiment_name, 'hypers')

    if input_space.num_dims != 1:
        raise Exception("This plotting script is only for 1D optimizations. This problem has %d dimensions." % input_space.num_dims)
    tasks = parse_tasks_from_jobs(jobs, experiment_name, options, input_space)

    hypers = chooser.fit(tasks, hypers)

    print '\nHypers:'
    print_hypers(hypers)

    recommendation = chooser.best()
    current_best_value = recommendation['model_model_value']
    current_best_location = recommendation['model_model_input']

    plots_dir = os.path.join(expt_dir, 'plots')
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
    if len(chooser.models) > 1:
        for task_name in chooser.models:
            plots_subdir = os.path.join(plots_dir, task_name)
            if not os.path.isdir(plots_subdir):
                os.mkdir(plots_subdir)

    print 'Plotting...'

    for task_name, model in chooser.models.iteritems():
        plots_subdir = os.path.join(plots_dir, task_name) if len(chooser.models) > 1 else plots_dir
        plot_mean_and_var(model, plots_subdir, input_space, current_best_location)
        plot_acquisition_function(model, chooser, plots_subdir, input_space, current_best_location, current_best_value)


# usage: python plots_1d.py DIRECTORY
if __name__ == '__main__':
    main(*sys.argv[1:])