import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils import utils
from pipes import flowmia_pipe as fmp

import argparse

import numpy as np
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Arquivo de configuração.")
    args = parser.parse_args()

    config = utils.read_config(args.config)

    ablation_result_folder = utils.validate_folder(config['io']['ablation_result_folder'])

    member_path = utils.validate_file(config['io']['real_train_dataset_input'])
    non_member_path = utils.validate_file(config['io']['reference_dataset_input'])
    test_path = utils.validate_file(config['io']['real_test_dataset_input'])

    flowmiagan_save_folder = utils.validate_folder(config['io']['flowmiagan_save_folder'])
    domias_save_folder = utils.validate_folder(config['io']['domias_save_folder'])
    dcr_save_folder = utils.validate_folder(config['io']['dcr_save_folder'])

    sample_tracker_file = utils.validate_file(config['io']['sample_tracker_file'])
    sample_dataset_folder = utils.validate_folder(config['io']['sample_dataset_folder'])

    start_test = 0
    end_test = float('inf')

    if 'start_test' in config['test']:
        start_test = config['test']['start_test']
    if 'end_test' in config['test']:
        end_test = config['test']['end_test']

    sample_tracker = pd.read_csv(sample_tracker_file)

    gen_mask = sample_tracker['test_idx'].between(start_test, end_test)
    sample_tracker = sample_tracker[gen_mask]

    aux_columns = ['test_idx', 'dataset_file']
    hyper_columns = sample_tracker.columns.drop(aux_columns).to_list()
    flowmiagan_columns = ['flowmiagan_auc']
    domias_columns = ['domias_auc']
    dcr_columns = ['dcr_auc']
    classification_columns = ['accuracy', 'precision', 'recall', 'f1-score']

    ablation_csv = ablation_result_folder / 'ablation.csv'

    if not ablation_csv.is_file() or ablation_csv.stat().st_size == 0:
        columns = ['test_idx'] + hyper_columns + flowmiagan_columns + domias_columns + dcr_columns + classification_columns
        ablation_df = pd.DataFrame(columns=columns)
    else:
        ablation_df = pd.read_csv(ablation_csv)

    for index, row in sample_tracker.iterrows():
        row_result = {}

        aux = row[aux_columns].to_dict()
        hyperparameters = row[hyper_columns].to_dict()

        row_result['test_idx'] = aux['test_idx']
        for i in hyper_columns:
            row_result[i] = hyperparameters[i]

        synth_path = sample_dataset_folder / aux['dataset_file']
        save_path = utils.validate_folder(flowmiagan_save_folder / f'test_{aux["test_idx"]}')

        flowmia_config = fmp.create_flowmia_config(member_path=member_path, 
                                                non_member_path=non_member_path,
                                                synth_path=synth_path,
                                                test_path=test_path,
                                                save_path=save_path)

        domias_save_path = utils.validate_folder(domias_save_folder / f'test_{aux["test_idx"]}')
        pipe_result = fmp.execute_pipe(flowmia_config, domias_save_path)

        scores_flowmiagan = pipe_result["flowmiagan"]
        scores_domias= pipe_result["domias"]
        scores_dcr= pipe_result["dcr"]
        utility_dict = pipe_result["utility_dict"]

        np.save(save_path / f'test_{aux["test_idx"]}.npy', scores_flowmiagan)

        np.save(domias_save_path / f'test_{aux["test_idx"]}.npy', scores_domias)

        np.save(dcr_save_folder / f'test_{aux["test_idx"]}.npy', scores_dcr)

        row_result['flowmiagan_auc'] = pipe_result['flowmiagan_auc']
        row_result['domias_auc'] = pipe_result['domias_auc']
        row_result['dcr_auc'] = pipe_result['dcr_auc']

        for key in utility_dict:
            row_result[key.lower()] = utility_dict[key]
        
        new_line = pd.DataFrame([row_result])
        ablation_df = pd.concat([ablation_df, new_line], ignore_index=True)
        ablation_df.to_csv(ablation_csv, index=False)