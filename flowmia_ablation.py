from utils import utils
from flowmia.flowmia import FlowMIA

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

def create_flowmia_config(member_path, non_member_path, synth_path, test_path, save_path):
    return {
        'member_path': member_path, # path dos membros
        'non_member_path': non_member_path, # path dos não-membros
        'synth_path': synth_path, # path dos sintéticos
        'test_path': test_path, # path do teste
        'categorical_cols': ['proto'], # colunas categóricas
        'numerical_cols': ['srcport', 'dstport', 'td', 'pkt', 'byt'], #colunas numéricas
        'ip_cols': ['srcip', 'dstip'], # colunas de ip
        'label_col': 'label', # nome da coluna do rótulo 
        'batch_size': 1000, # número de amostrar por lote
        'num_epochs': 50, # número de épocas
        'fcheckpoint': 100, # frequência para salvar o checkpoint
        'save_path': save_path,
        'use_wgan': True, # se deve usar WGAN ou GAN tradicional
        'test_size': 10000
    }

def validate_file(file_path):
    file = Path(file_path)
    if not file.is_file():
        raise Exception(f"Arquivo {file} não encontrado.")
    return file

def validate_folder(folder_path, create=True):
    folder = Path(folder_path)
    if not folder.is_dir():
        if create:
            folder.mkdir(parents=True, exist_ok=True)
        else:
            raise Exception(f"Pasta {folder} não encontrada.")
    return folder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Arquivo de configuração.")
    args = parser.parse_args()

    config = utils.read_config(args.config)

    ablation_result_folder = validate_folder(config['io']['ablation_result_folder'])

    member_path = validate_file(config['io']['real_train_dataset_input'])
    non_member_path = validate_file(config['io']['reference_dataset_input'])
    test_path = validate_file(config['io']['real_test_dataset_input'])

    flowmiagan_save_folder = validate_folder(config['io']['flowmiagan_save_folder'])
    domias_save_folder = validate_folder(config['io']['domias_save_folder'])
    dcr_save_folder = validate_folder(config['io']['dcr_save_folder'])

    sample_tracker_file = validate_file(config['io']['sample_tracker_file'])
    sample_dataset_folder = validate_folder(config['io']['sample_dataset_folder'])

    start_test = 0
    end_test = float('inf')

    if 'start_test' in config['test']:
        start_test = config['test']['start_test']
    if 'end_test' in config['test']:
        end_test = config['test']['end_test']

    generate_tracker = pd.read_csv(sample_tracker_file)

    gen_mask = generate_tracker['test_idx'].between(start_test, end_test)
    generate_tracker = generate_tracker[gen_mask]

    aux_columns = ['test_idx', 'dataset_file']
    hyper_columns = generate_tracker.columns.drop(aux_columns).to_list()
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

    for index, row in generate_tracker.iterrows():
        row_result = {}

        aux = row[aux_columns].to_dict()
        hyperparameters = row[hyper_columns].to_dict()

        row_result['test_idx'] = aux['test_idx']
        for i in hyper_columns:
            row_result[i] = hyperparameters[i]

        synth_path = sample_dataset_folder / aux['dataset_file']
        save_path = validate_folder(flowmiagan_save_folder / f'test_{aux["test_idx"]}')

        flowmia_config = create_flowmia_config(member_path=member_path, 
                                                non_member_path=non_member_path,
                                                synth_path=synth_path,
                                                test_path=test_path,
                                                save_path=save_path)

        flowmia = FlowMIA(config=flowmia_config)

        scores_flowmiagan = flowmia.flowmiagan(test_size = flowmia_config['test_size'])
        np.save(save_path / f'test_{aux["test_idx"]}.npy', scores_flowmiagan)

        domias_save_path = validate_folder(domias_save_folder / f'test_{aux["test_idx"]}')
        scores_domias, auc_domias = flowmia.domias(test_size=flowmia_config['test_size'], save_path=domias_save_path, epochs=30)
        np.save(domias_save_path / f'test_{aux["test_idx"]}.npy', scores_domias)

        scores_dcr, auc_dcr = flowmia.compute_dcr(test_size=flowmia_config['test_size'])
        np.save(dcr_save_folder / f'test_{aux["test_idx"]}.npy', scores_dcr)


        flowmiagan_scores_aux = np.concatenate([scores_flowmiagan['score_members'], scores_flowmiagan['score_non_members']])
        y_test = np.concatenate([np.ones(flowmia_config['test_size']), np.zeros(flowmia_config['test_size'])])

        row_result['flowmiagan_auc'] = roc_auc_score(y_test, flowmiagan_scores_aux)
        row_result['domias_auc'] = roc_auc_score(y_test, scores_domias)
        row_result['dcr_auc'] = roc_auc_score(y_test, scores_dcr)

        classifiers = [ RandomForestClassifier(n_estimators=100, random_state=42) ]
        utility_dict = flowmia.evaluate_utility(classifiers=classifiers)['RandomForestClassifier']['TSTR']

        for key in utility_dict:
            row_result[key.lower()] = utility_dict[key]
        
        new_line = pd.DataFrame([row_result])
        ablation_df = pd.concat([ablation_df, new_line], ignore_index=True)
        ablation_df.to_csv(ablation_csv, index=False)