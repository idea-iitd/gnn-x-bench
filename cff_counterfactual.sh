#!/usr/bin/bash

device=0
dataset=Proteins
python source/cff.py --alp 0.0 --device $device --explainer_run 1 --gnn_type gcn --dataset $dataset
python source/cff.py --alp 0.0 --device $device --explainer_run 2 --gnn_type gcn --dataset $dataset
python source/cff.py --alp 0.0 --device $device --explainer_run 3 --gnn_type gcn --dataset $dataset
python source/cff.py --alp 0.0 --device $device --explainer_run 1 --gnn_type gin --dataset $dataset
python source/cff.py --alp 0.0 --device $device --explainer_run 1 --gnn_type gat --dataset $dataset
python source/cff.py --alp 0.0 --device $device --explainer_run 1 --gnn_type sage --dataset $dataset
python source/cff.py --alp 0.0 --device $device --explainer_run 1 --gnn_type gcn --dataset $dataset --robustness
