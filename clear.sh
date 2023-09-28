#!/usr/bin/bash
cd source/methods/clear/src

datasets=("Mutag" "IMDB-B" "AIDS") # other datasets give OOM error.
for dataset in "${datasets[@]}"
do
    python main.py -e train -d $dataset --device 0 --save_model --save_result --epochs 1200 --layer gin --seed 1
    python main.py -e train -d $dataset --device 0 --save_model --save_result --epochs 1200 --layer gat --seed 1
    python main.py -e train -d $dataset --device 0 --save_model --save_result --epochs 1200 --layer sage --seed 1
    python main.py -e train -d $dataset --device 0 --save_model --save_result --epochs 1200 --layer gcn --seed 1
    python main.py -e train -d $dataset --device 0 --save_model --save_result --epochs 1200 --layer gcn --seed 2
    python main.py -e train -d $dataset --device 0 --save_model --save_result --epochs 1200 --layer gcn --seed 3
    python main.py -e train -d $dataset --device 0 --save_model --save_result --epochs 1200 --layer gcn --seed 1 --robustness
done

datasets=("Mutag" "IMDB-B" "AIDS") # other datasets give OOM error.
for dataset in "${datasets[@]}"
do
    python main.py -e test -d $dataset --device 0 --seed 1 --layer gin
    python main.py -e test -d $dataset --device 0 --seed 1 --layer gat
    python main.py -e test -d $dataset --device 0 --seed 1 --layer sage
    python main.py -e test -d $dataset --device 0 --seed 1 --layer gcn
    python main.py -e test -d $dataset --device 0 --seed 2 --layer gcn
    python main.py -e test -d $dataset --device 0 --seed 3 --layer gcn
    python main.py -e test -d $dataset --device 0 --seed 1 --layer gcn --robustness
done
