gnn_type=gcn
device=1
cpus=0-15

# collect explanations from datasets
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset Mutagenicity
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset Proteins
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset IMDB-B
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset AIDS
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset Mutag
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset NCI1
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset Graph-SST2
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset REDDIT-B
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset DD
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset ogbg_molhiv

# collect explanations from noisy datasets
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --robustness topology_random --lambda_ 1.0 --dataset Mutagenicity
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --robustness topology_random --lambda_ 1.0 --dataset Proteins
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --robustness topology_random --lambda_ 1.0 --dataset IMDB-B
taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --robustness topology_random --lambda_ 1.0 --dataset AIDS

# stability seeds
seeds="2 3"
for seed in $seeds; do
  taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --explainer_run $seed --lambda_ 1.0 --dataset Mutagenicity
  taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --explainer_run $seed --lambda_ 1.0 --dataset Proteins
  taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --explainer_run $seed --lambda_ 1.0 --dataset IMDB-B
  taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --explainer_run $seed --lambda_ 1.0 --dataset AIDS
  taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --explainer_run $seed --lambda_ 1.0 --dataset Mutag
  taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --explainer_run $seed --lambda_ 1.0 --dataset NCI1
done

# stability base
gnn_types="gat gin sage"
for gnn_type in $gnn_types; do
  taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset Mutagenicity
  taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset Proteins
  taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset IMDB-B
  taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset AIDS
  taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset Mutag
  taskset -c $cpus python source/rcexplainer.py --gnn_type $gnn_type --device $device --lambda_ 1.0 --dataset NCI1
done

# generate results
#gnn_type=gcn
#explainers="rcexplainer_1.0"
#metrics="faithfulness faithfulness_with_removal stability_noise stability_seed stability_base"
#for metric in $metrics; do
#  for explainer in $explainers; do
#    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutagenicity --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
#    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Proteins --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
#    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset IMDB-B --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
#    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset AIDS --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
#    if ! (("$metric" = "stability_noise")); then
#      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutag --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
#      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset NCI1 --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
#    fi
#    if [ "$metric" = "faithfulness" ]; then
#      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Graph-SST2 --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric"
#      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset ogbg_molhiv -gnn_type "$gnn_type" --device cpu --explanation_metric "$metric"
#      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset REDDIT-B --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric"
#      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset DD --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric"
#    fi
#  done
#done
