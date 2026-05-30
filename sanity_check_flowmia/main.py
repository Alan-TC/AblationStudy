import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils import utils
from pipes import flowmia_pipe as fp

import argparse

import numpy as np
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Arquivo de configuração.")
    args = parser.parse_args()

    config = utils.read_config(args.config)

    result_folder = utils.validate_folder(config['io']['result_folder'])

    member_path = utils.validate_file(config['io']['real_train_dataset_input'])
    non_member_path = utils.validate_file(config['io']['reference_dataset_input'])
    test_path = utils.validate_file(config['io']['real_test_dataset_input'])
    synthetic_path = utils.validate_file(config['io']['synthetic_dataset_input'])


    seeds_test = config['test']['seeds']
    num_epochs = config['test']['num_epochs']

    flowmiagan_columns = ['flowmiagan_auc']
    domias_columns = ['domias_auc']
    dcr_columns = ['dcr_auc']

    result_csv = result_folder / 'result.csv'

    columns = ['test_id', 'num_epochs', 'seed'] + flowmiagan_columns + domias_columns + dcr_columns

    result_df = pd.DataFrame(columns=columns)

    test_id = 0

    # O DCR é sempre o mesmo nos testes
    dcr_save_folder = utils.validate_folder(config['io']['dcr_save_folder'])

    config_dcr = fp.create_flowmia_config(
        member_path=member_path, 
        non_member_path=non_member_path, 
        synth_path=synthetic_path, 
        test_path=test_path)

    scores_dcr, auc_dcr = fp.execute_dcr(config_dcr)
    np.save(dcr_save_folder / f'scores.npy', scores_dcr)

    for seed in seeds_test:
        for num_epoch in num_epochs:
            test_id += 1
            test_name = f'test_{test_id}'

            flowmiagan_save_folder = utils.validate_folder(config['io']['flowmiagan_save_folder'])
            flowmiagan_save_folder = utils.validate_folder(flowmiagan_save_folder / test_name)

            domias_save_folder = utils.validate_folder(config['io']['domias_save_folder'])
            domias_save_folder = utils.validate_folder(domias_save_folder / test_name)

            flowmia_config = fp.create_flowmia_config(member_path=member_path, 
                                                    non_member_path=non_member_path,
                                                    synth_path=synthetic_path,
                                                    test_path=test_path,
                                                    save_path=flowmiagan_save_folder,
                                                    num_epochs=num_epoch,
                                                    random_seed=seed)

            scores_flowmiagan, auc_flowmiagan = fp.execute_flowmia(flowmia_config)
            np.save(flowmiagan_save_folder / f'scores.npy', scores_flowmiagan)

            scores_domias, auc_domias = fp.execute_domias(flowmia_config, domias_save_folder, epochs=num_epoch)

            np.save(domias_save_folder / f'scores.npy', scores_domias)

            row_result = {}
            row_result['test_id'] = test_id
            row_result['num_epochs'] = num_epoch
            row_result['seed'] = seed
            row_result['flowmiagan_auc'] = auc_flowmiagan
            row_result['domias_auc'] = auc_domias
            row_result['dcr_auc'] = auc_dcr

            new_line = pd.DataFrame([row_result])
            result_df = pd.concat([result_df, new_line], ignore_index=True)
            result_df.to_csv(result_csv, index=False)