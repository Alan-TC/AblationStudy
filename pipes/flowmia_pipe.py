import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils import utils
from flowmia.flowmia import FlowMIA

import argparse

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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
        'test_size': 10000,
        'seed': 42
    }

def execute_pipe(flowmia_config, domias_save_path):
    result_dict = {}

    flowmia = FlowMIA(config=flowmia_config)

    result_dict["flowmiagan"] = flowmia.flowmiagan(test_size = flowmia_config['test_size'])

    result_dict["domias"] = flowmia.domias(test_size=flowmia_config['test_size'], save_path=domias_save_path, epochs=30)

    result_dict["dcr"] = flowmia.compute_dcr(test_size=flowmia_config['test_size'])

    scores_flowmiagan = result_dict["flowmiagan"]
    scores_domias, _ = result_dict["domias"] # (scores, auc)
    scores_dcr, _ = result_dict["dcr"] # (scores, auc)

    flowmiagan_scores_aux = np.concatenate([scores_flowmiagan['score_members'], scores_flowmiagan['score_non_members']])
    y_test = np.concatenate([np.ones(flowmia_config['test_size']), np.zeros(flowmia_config['test_size'])])

    result_dict['flowmiagan_auc'] = roc_auc_score(y_test, flowmiagan_scores_aux)
    result_dict['domias_auc'] = roc_auc_score(y_test, scores_domias)
    result_dict['dcr_auc'] = roc_auc_score(y_test, scores_dcr)

    classifiers = [ RandomForestClassifier(n_estimators=100, random_state=42) ]
    result_dict["utility_dict"] = flowmia.evaluate_utility(classifiers=classifiers)['RandomForestClassifier']['TSTR']

    return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Arquivo de configuração.")
    args = parser.parse_args()

    config = utils.read_config(args.config)

    result_folder = utils.validate_folder(config['io']['result_folder'])

    member_path = utils.validate_file(config['io']['real_train_dataset_input'])
    non_member_path = utils.validate_file(config['io']['reference_dataset_input'])
    test_path = utils.validate_file(config['io']['real_test_dataset_input'])
    synthetic_path = utils.validate_file(config['io']['synthetic_dataset_input'])

    flowmiagan_save_folder = utils.validate_folder(config['io']['flowmiagan_save_folder'])
    domias_save_folder = utils.validate_folder(config['io']['domias_save_folder'])
    dcr_save_folder = utils.validate_folder(config['io']['dcr_save_folder'])

    flowmia_config = create_flowmia_config(member_path=member_path, 
                                            non_member_path=non_member_path,
                                            synth_path=synthetic_path,
                                            test_path=test_path,
                                            save_path=flowmiagan_save_folder)

    pipe_result = execute_pipe(flowmia_config, domias_save_folder)

    scores_flowmiagan = pipe_result["flowmiagan"]
    scores_domias, auc_domias = pipe_result["domias"]
    scores_dcr, auc_dcr = pipe_result["dcr"]
    utility_dict = pipe_result["utility_dict"]

    np.save(flowmiagan_save_folder / f'scores.npy', scores_flowmiagan)

    np.save(domias_save_folder / f'scores.npy', scores_domias)

    np.save(dcr_save_folder / f'scores.npy', scores_dcr)

    flowmiagan_columns = ['flowmiagan_auc']
    domias_columns = ['domias_auc']
    dcr_columns = ['dcr_auc']
    classification_columns = ['accuracy', 'precision', 'recall', 'f1-score']

    result_csv = result_folder / 'result.csv'

    columns = flowmiagan_columns + domias_columns + dcr_columns + classification_columns

    row_result = {}

    row_result['flowmiagan_auc'] = pipe_result['flowmiagan_auc']
    row_result['domias_auc'] = pipe_result['domias_auc']
    row_result['dcr_auc'] = pipe_result['dcr_auc']

    for key in utility_dict:
        row_result[key.lower()] = utility_dict[key]
    
    result_df = pd.DataFrame([row_result])
    result_df.to_csv(result_csv, index=False)