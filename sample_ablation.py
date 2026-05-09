from utils import utils

import csv
import argparse
from pathlib import Path

import pandas as pd
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.serialization import save_to_file


def create_plugin_config(n_iter, batch_size, n_units_hidden):
    return dict(
            is_classification = True,
            n_iter = n_iter,  # epochs 50, 100, 150, 200
            lr = 0.002,
            weight_decay = 1e-4,
            batch_size = batch_size, # 50, 100, 500, 1000
            model_type = "mlp",  # or "resnet"
            model_params = dict(
                n_layers_hidden = 2,
                n_units_hidden = n_units_hidden, # 32, 64, 128
                dropout = 0.0,
            ),
            num_timesteps = 100,  # timesteps in diffusion
            dim_embed = 128,
            log_interval = 10
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Caminho para o arquivo de configuração.")

    args = parser.parse_args()

    config = utils.read_config(args.config)
    
    real_dataset_input = Path(config['io']['real_dataset_input'])
    print(real_dataset_input)
    if not real_dataset_input.is_file():
        raise Exception(f"Arquivo {config['io']['real_dataset_input']} não encontrado.")

    tracker_output_folder = Path(config['io']['tracker_file_output_folder'])
    tracker_output_folder.mkdir(parents=True, exist_ok=True)

    model_output_folder = Path(config['io']['results_output_model_folder'])
    model_output_folder.mkdir(parents=True, exist_ok=True)
    
    dataset_output_folder = Path(config['io']['results_output_sample_folder'])
    dataset_output_folder.mkdir(parents=True, exist_ok=True)
    
    start_test = 0
    end_test = float('inf')
    sample_len = config['test']['sample_len']

    if 'start_test' in config['test']:
        start_test = config['test']['start_test']
    if 'end_test' in config['test']:
        end_test = config['test']['end_test']

    hyperparameters = config['hyperparameters']
    
    print('Configuration Finished')


    X = pd.read_csv(real_dataset_input)

    loader = GenericDataLoader(X, target_column="label", sensitive_columns=[])
    print('DataLodader Created')


    generate_tracker_path = tracker_output_folder / 'generate_tracker.csv'

    with open(generate_tracker_path, 'a', encoding='utf-8', newline='') as f:
        if f.tell() == 0:
            csv.writer(f).writerow(['test_idx', 'n_iter', 'batch_size', 'n_units_hidden' , 'dataset_file'])


    curr_test = 0

    for n_iter in hyperparameters['n_iter']:
        for batch_size in hyperparameters['batch_size']:
            for n_units_hidden in hyperparameters['n_units_hidden']:

                curr_test += 1
                jump_this_test = not (start_test <= curr_test <= end_test)
                if jump_this_test:
                    continue
                
                print()
                print(curr_test, ' - ', 'n_iter:', n_iter, ' | batch_size:', batch_size, '| n_units_hidden:', n_units_hidden)

                plugin_params = create_plugin_config(n_iter=n_iter, 
                                                     batch_size=batch_size, 
                                                     n_units_hidden=n_units_hidden)

                plugin = Plugins().get("ddpm", **plugin_params)

                plugin.fit(loader)

                X_synthetic = plugin.generate(sample_len)

                dataset_output_file = f'test_{curr_test}.csv'
                model_output_file = f'test_{curr_test}.pkl'

                X_synthetic.dataframe().to_csv(dataset_output_folder / dataset_output_file, index=False)
                save_to_file(model_output_folder / model_output_file, plugin)

                with open(generate_tracker_path, 'a', encoding='utf-8', newline='') as f:
                    csv.writer(f).writerow([curr_test, n_iter, batch_size, n_units_hidden, f"{dataset_output_file}"])