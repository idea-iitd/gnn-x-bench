gnn_type=gcn
device=0
cpus=0-15

# collect explanations from datasets
taskset -c $cpus python source/gnnexplainer.py --dataset Mutagenicity --gnn_type $gnn_type --device $device
taskset -c $cpus python source/gnnexplainer.py --dataset Proteins --gnn_type $gnn_type --device $device
taskset -c $cpus python source/gnnexplainer.py --dataset IMDB-B --gnn_type $gnn_type --device $device
taskset -c $cpus python source/gnnexplainer.py --dataset AIDS --gnn_type $gnn_type --device $device
taskset -c $cpus python source/gnnexplainer.py --dataset Mutag --gnn_type $gnn_type --device $device
taskset -c $cpus python source/gnnexplainer.py --dataset NCI1 --gnn_type $gnn_type --device $device
taskset -c $cpus python source/gnnexplainer.py --dataset Graph-SST2 --gnn_type $gnn_type --device $device
taskset -c $cpus python source/gnnexplainer.py --dataset REDDIT-B --gnn_type $gnn_type --device $device
taskset -c $cpus python source/gnnexplainer.py --dataset DD --gnn_type $gnn_type --device $device

# collect explanations from noisy datasets
taskset -c $cpus python source/gnnexplainer.py --dataset Mutagenicity --gnn_type $gnn_type --device $device --robustness
taskset -c $cpus python source/gnnexplainer.py --dataset Proteins --gnn_type $gnn_type --device $device --robustness
taskset -c $cpus python source/gnnexplainer.py --dataset IMDB-B --gnn_type $gnn_type --device $device --robustness
taskset -c $cpus python source/gnnexplainer.py --dataset AIDS --gnn_type $gnn_type --device $device --robustness

# stability seeds
seeds="2 3"
for seed in $seeds; do
  taskset -c $cpus python source/gnnexplainer.py --dataset Mutagenicity --gnn_type $gnn_type --device $device --explainer_run $seed
  taskset -c $cpus python source/gnnexplainer.py --dataset Proteins --gnn_type $gnn_type --device $device --explainer_run $seed
  taskset -c $cpus python source/gnnexplainer.py --dataset IMDB-B --gnn_type $gnn_type --device $device --explainer_run $seed
  taskset -c $cpus python source/gnnexplainer.py --dataset AIDS --gnn_type $gnn_type --device $device --explainer_run $seed
  taskset -c $cpus python source/gnnexplainer.py --dataset Mutag --gnn_type $gnn_type --device $device --explainer_run $seed
  taskset -c $cpus python source/gnnexplainer.py --dataset NCI1 --gnn_type $gnn_type --device $device --explainer_run $seed
done

# stability base
gnn_types="gat gin sage"
for gnn_type in $gnn_types; do
  taskset -c $cpus python source/gnnexplainer.py --dataset Mutagenicity --gnn_type $gnn_type --device $device
  taskset -c $cpus python source/gnnexplainer.py --dataset Proteins --gnn_type $gnn_type --device $device
  taskset -c $cpus python source/gnnexplainer.py --dataset IMDB-B --gnn_type $gnn_type --device $device
  taskset -c $cpus python source/gnnexplainer.py --dataset AIDS --gnn_type $gnn_type --device $device
  taskset -c $cpus python source/gnnexplainer.py --dataset Mutag --gnn_type $gnn_type --device $device
  taskset -c $cpus python source/gnnexplainer.py --dataset NCI1 --gnn_type $gnn_type --device $device
done

# generate results
#gnn_type=gcn
#explainers="gnnexplainer"
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
#      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset REDDIT-B --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric"
#      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset DD --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric"
#    fi
#  done
#done
