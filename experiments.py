from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from EasyTSAD.Controller import TSADController
from EasyTSAD.Methods import LSTMADalpha, LSTMADbeta
from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
import gc
import os
import shutil
import json
import pandas as pd
import toml
import copy 
 

# Import your own algo first before using it
from LSTMADalpha_resilientbackprop import LSTMADalpha_resilientbackprop

EXPERIMENT = 3

def run_only_once(gctrl, methods, training_schema):
    """============= [EXPERIMENTAL SETTINGS] ============="""
    for i, method in enumerate(methods):
        # run models with default config
        gctrl.run_exps(
            method=method,
            training_schema=training_schema,
        )

def hyperparameter_tuning(gctrl, method, default_hparams, param_name, param_range : tuple, training_schema): 
    methods = []
    param_range = np.linspace(*param_range) 
    hparams = [{} for _ in param_range]
    for i, param_value in enumerate(param_range):

        hparams[i] = copy.deepcopy(default_hparams)
        param_value = int(param_value)
        print(f'Training the model with {param_value} {param_name}')
        hparams[i][param_name] = param_value
        
        gctrl.run_exps(
            method=method,
            training_schema=training_schema,
            hparams = hparams[i]
        )
        new_method_name = f'{method}-{param_name}-{i}'

        dirs_to_rename = ['Evals', 'Plots', 'RunTime', 'Scores']
        for dir in dirs_to_rename:
            path = os.path.join('Results', dir, method)
            if os.path.exists(path):
                new_path = os.path.join('Results', dir, new_method_name)
                os.rename(path, new_path)

        methods.append(new_method_name)
        gc.collect()
    
    return methods, hparams

if __name__ == "__main__":
    # if cfg_path is None, using default configuration
    gctrl = TSADController()
    dirname = './datasets/'

    # Select dataset
    datasets = ["AIOPS"]

    # Adjust the curves on which the model will be trained
    gctrl.set_dataset(
        dataset_type="UTS",
        dirname=dirname,
        datasets='AIOPS',
        specify_curves=True,
        curve_names=[
            "0efb375b-b902-3661-ab23-9a0bb799f4e3"
            # "ab216663-dcc2-3a24-b1ee-2c3e550e06c9"
        ]
    )

    training_schema = "naive"

    # Read default hparams
    with open('./default_hparams.toml', 'r') as f:
        config = toml.load(f)

    default_hparams = config['Model_Params']['Default']

    print(f'============= [PREPARATION SECTION] =============')
    dirs_to_del = ['Evals', 'Plots', 'RunTime', 'Scores']
    for dir in dirs_to_del:
        if os.path.exists(f'./Results/{dir}'):
            shutil.rmtree(f'./Results/{dir}')
    print('Successfully cleaned the Results directory.')

    print(f'============= [EXPERIMENT {EXPERIMENT} SECTION] =============')

    methods = []

    # carry out evals
    if EXPERIMENT == 0:
        # EXPERIMENT 0: Run the baseline models 
        # Run baseline methods
        methods = ["LSTMADalpha", "LSTMADbeta"]
        # If your have run this function and don't change the params, you can skip this step.
        run_only_once(gctrl, methods, training_schema)

    elif EXPERIMENT == 1: 
        # EXPERIMENT 1: Run the plain method LSTMADbeta first and then run the custom method using resilient backprop
        methods = ['LSTMADalpha', 'LSTMADalpha_resilientbackprop']
        method = methods[0]
        gctrl.run_exps(
            method=method,
            training_schema=training_schema,
        )
        # name of the custom algorithm
        method = methods[1]
        # run with config 
        # cfg_path = f'./{method}/config.toml'
        # Run with custom config file
        gctrl.run_exps(
            method=method,
            training_schema=training_schema,
            hparams=default_hparams
        )
        hparams = [default_hparams for i in range(len(methods))]

    elif EXPERIMENT == 2:
        # Hyperparameter Tuning with Adam Optimizer
        method = 'LSTMADalpha'

        # hidden size
        hidden_range = (10,50,5)
        methods, hparams = hyperparameter_tuning(gctrl, method, default_hparams, 'hidden_dim', hidden_range, training_schema)
        # num layer
        num_layer_range = (1,5,5)
        methods_2, hparams_2 = hyperparameter_tuning(gctrl, method, default_hparams, 'num_layer', num_layer_range, training_schema)
        # prediction length
        pred_len_range = (1,5,5)
        methods_3, hparams_3 = hyperparameter_tuning(gctrl, method, default_hparams, 'pred_len', pred_len_range, training_schema)

        methods = methods + methods_2 + methods_3
        hparams = hparams + hparams_2 + hparams_3

    elif EXPERIMENT == 3:
        # Hyperparameter Tuning with Resilient Optimizer
        method = 'LSTMADalpha_resilientbackprop'

        # hidden size
        hidden_range = (10,50,5)
        methods, hparams = hyperparameter_tuning(gctrl, method, default_hparams, 'hidden_dim', hidden_range, training_schema)
        # num layer
        num_layer_range = (1,5,5)
        methods_2, hparams_2 = hyperparameter_tuning(gctrl, method, default_hparams, 'num_layer', num_layer_range, training_schema)
        # prediction length
        pred_len_range = (1,5,5)
        methods_3, hparams_3 = hyperparameter_tuning(gctrl, method, default_hparams, 'pred_len', pred_len_range, training_schema)

        methods = methods + methods_2 + methods_3
        hparams = hparams + hparams_2 + hparams_3

    else:
        print(f'The provided experiment {EXPERIMENT} is not available.')

    print("============= [EVALUATION SECTION] =============")
    
    # Specifying evaluation protocols
    gctrl.set_evals(
        [
            PointF1PA(),
            EventF1PA(),
            EventF1PA(mode="squeeze")
        ]
    )

    for method in methods:
        gctrl.do_evals(
            method=method,
            training_schema=training_schema
        )
        
        

    print("============= [REPORT SECTION] =============")

    report = []

    for i, method in enumerate(methods): 
        for dataset in datasets:
            base_path = 'Results/Evals'
            results_filename = 'avg.json'
            results_path = os.path.join(base_path, method, training_schema, dataset,results_filename)
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f: 
                    results_dict = json.load(f)
                f1_score = results_dict['best f1 under pa']['f1']
            else:
                raise FileNotFoundError(f'Results file {results_path} not found.')
            base_path = 'Results/RunTime'
            time_filename = 'time.json'
            time_path = os.path.join(base_path, method, training_schema, dataset,time_filename)
            if os.path.exists(time_path):
                with open(time_path, 'rb') as f: 
                    results_dict = json.load(f)
                train_time = results_dict['train_and_valid']
                test_time = results_dict['test']
            else:
                raise FileNotFoundError(f'Runtime file {time_path} not found.')
            method_report = {
                "Method":method,
                "Dataset":dataset,
                "F1": f'{f1_score:.2f}',
                "Training time [s]": f'{train_time:.2f}',
                "Test time [s]": f'{test_time:.2f}'
            }
            method_report.update(hparams[i])
            report.append(method_report)
    report_df = pd.DataFrame(report)
    report_df.reset_index()
    report_df.set_index('Method')
    report_df.astype({"hidden_dim":int, "pred_len":int, "num_layer":int})

    print(report_df)

    report_df.to_excel(f'Experiment_{EXPERIMENT}_Results.xlsx')


    print("============= [PLOTTING SECTION] =============")
    for method in methods: 
        # plot anomaly scores for each curve
        gctrl.plots(
            method=method,
            training_schema=training_schema
        )
    if EXPERIMENT == 2 or EXPERIMENT == 3:
        def plot_subplot(ax, data, filter_column, x_label, x_dtype='int16'):
            """
            Plot a subplot with dual y-axes for F1 Score and Training Time.
            
            Parameters:
            ax : matplotlib.axes.Axes
                The subplot axis to plot on.
            data : pandas.DataFrame
                The DataFrame containing the report data.
            filter_column : str
                The column name substring to filter the DataFrame.
            x_label : str
                Label for the x-axis.
            x_dtype : str
                Data type to cast the x-axis data to (default is 'int16').
            """
            filtered_df = data[data['Method'].str.contains(filter_column)]
            x = filtered_df[filter_column].astype(x_dtype)
            f1_scores = filtered_df['F1']
            training_time = filtered_df['Training time [s]']
            
            ax.plot(x, f1_scores, 'b-', label='F1 Score [%]')
            ax.set_xlabel(x_label)
            ax.set_ylabel('F1 Score [%]', color='b')
            ax.tick_params(axis='y', labelcolor='b')

            ax_b = ax.twinx()
            ax_b.plot(x, training_time, 'r-', label='Training Time [s]')
            ax_b.set_ylabel('Training Time [s]', color='r')
            ax_b.tick_params(axis='y', labelcolor='r')

        def plot_performance_comparison(report_df, experiment):
            """
            Plot performance and training time comparison for different hyperparameters.
            
            Parameters:
            report_df : pandas.DataFrame
                The DataFrame containing the report data.
            experiment : int
                Experiment identifier to select the optimizer name.
            """
            fig, axs = plt.subplots(nrows=3, figsize=(10, 10))
            
            optimizer = 'ADAM optimizer' if experiment == 2 else 'Resilient Backpropagation'
            title = f"Hyperparameter tuning of LSTMADs using {optimizer}\n\nPerformance and Training Time Comparison"
            plt.suptitle(title, y=0.99, fontsize=16)
            
            plots_config = [
                {'filter_column': 'hidden_dim', 'x_label': 'Number of hidden dimensions'},
                {'filter_column': 'num_layer', 'x_label': 'Number of LSTM layers'},
                {'filter_column': 'pred_len', 'x_label': 'Number of predicted time steps'}
            ]
            
            for ax, config in zip(axs, plots_config):
                plot_subplot(ax, report_df, config['filter_column'], config['x_label'])
            
            plt.tight_layout()
            plt.savefig(f'Experiment_{experiment}_Results.png')

        plot_performance_comparison(report_df, EXPERIMENT)

    print("============= [CLEANING SECTION] =============")

    # Clean up the RAM
    del gctrl
    unreachable_objects = gc.collect()
    print(f'Number of deleted items: {unreachable_objects}')