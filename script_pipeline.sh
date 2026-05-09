#!/bin/bash

# Interrompe o script se houver erro em qualquer comando
set -e

echo "Iniciando experimento parte 1: gerando os samples"
conda run --no-capture-output -n AblationTestSynthcity python sample_ablation.py --config ablation/cidds/samples/config.toml

echo "Iniciando experimento parte 2: avaliando os conjuntos"
conda run --no-capture-output -n AblationTestFlowMia python flowmia_ablation.py --config ablation/cidds/results/config.toml

echo "Todos os testes foram concluídos com sucesso!"