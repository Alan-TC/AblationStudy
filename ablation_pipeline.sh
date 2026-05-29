#!/bin/bash

# Interrompe o script se houver erro em qualquer comando
set -e

DATASET_NAME=${1:-cidds}

# Verifica se o diretório base existe antes de tentar rodar
if [ ! -d "ablation/$DATASET_NAME" ]; then
    echo "Erro: O diretório 'ablation/$DATASET_NAME' não foi encontrado."
    exit 1
fi

echo "Iniciando experimento parte 1: gerando os samples ($DATASET_NAME)"
conda run --no-capture-output -n AblationTestSynthcity python ./ablation_pipes/sample_ablation.py --config ablation/"$DATASET_NAME"/samples/config.toml

echo "Iniciando experimento parte 2: avaliando os conjuntos ($DATASET_NAME)"
conda run --no-capture-output -n AblationTestFlowMia python ./ablation_pipes/flowmia_ablation.py --config ablation/"$DATASET_NAME"/results/config.toml

echo "Todos os testes foram concluídos com sucesso!"