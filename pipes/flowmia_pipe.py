from flowmia.flowmia import FlowMIA

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def create_flowmia_config(member_path, non_member_path, synth_path, test_path, save_path='.', num_epochs = 50, random_seed = 42):
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
        'num_epochs': num_epochs, # número de épocas
        'fcheckpoint': 100, # frequência para salvar o checkpoint
        'save_path': save_path,
        'use_wgan': True, # se deve usar WGAN ou GAN tradicional
        'test_size': 10000,
        'seed': random_seed
    }

def execute_flowmia(flowmia_config):
    flowmia = FlowMIA(config=flowmia_config)

    scores = flowmia.flowmiagan(test_size = flowmia_config['test_size'])

    flowmiagan_scores_aux = np.concatenate([scores['score_members'], scores['score_non_members']])
    y_test = np.concatenate([np.ones(flowmia_config['test_size']), np.zeros(flowmia_config['test_size'])])

    auc = roc_auc_score(y_test, flowmiagan_scores_aux)

    return scores, auc

def execute_domias(flowmia_config, domias_save_path):
    flowmia = FlowMIA(config=flowmia_config)

    scores, _ = flowmia.domias(test_size=flowmia_config['test_size'], save_path=domias_save_path, epochs=30)

    y_test = np.concatenate([np.ones(flowmia_config['test_size']), np.zeros(flowmia_config['test_size'])])
    auc = roc_auc_score(y_test, scores)

    return scores, auc

def execute_dcr(flowmia_config):
    flowmia = FlowMIA(config=flowmia_config)

    scores, _ = flowmia.compute_dcr(test_size=flowmia_config['test_size'])

    y_test = np.concatenate([np.ones(flowmia_config['test_size']), np.zeros(flowmia_config['test_size'])])
    auc = roc_auc_score(y_test, scores)

    return scores, auc

def execute_utility(flowmia_config):
    flowmia = FlowMIA(config=flowmia_config)

    classifiers = [ RandomForestClassifier(n_estimators=100, random_state=42) ]

    utility_dict = flowmia.evaluate_utility(classifiers=classifiers)['RandomForestClassifier']['TSTR']

    return utility_dict


def execute_pipe(flowmia_config, domias_save_path):
    result_dict = {}

    flowmia = FlowMIA(config=flowmia_config)

    result_dict["flowmiagan"], result_dict['flowmiagan_auc'] = execute_flowmia(flowmia_config)

    result_dict["domias"], result_dict['domias_auc'] = execute_domias(flowmia_config, domias_save_path)

    result_dict["dcr"], result_dict['dcr_auc'] = execute_dcr(flowmia_config)

    result_dict["utility_dict"] = execute_utility(flowmia_config)

    return result_dict