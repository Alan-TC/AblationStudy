#!/bin/bash

# Interrompe o script se houver erro em qualquer comando
set -e

echo "Sanity Teste 1: os dados de ton.csv como referência"
conda run --no-capture-output -n AblationTestFlowMia python ./sanity_check_flowmia/main.py --config sanity_check_flowmia/ton_as_reference/config.toml

echo "Sanity Teste 1: os dados de teste como referência"
conda run --no-capture-output -n AblationTestFlowMia python ./sanity_check_flowmia/main.py --config sanity_check_flowmia/test_as_reference/config.toml

