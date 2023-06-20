gnn_type=gcn
device=0
cpus=0-31

# collect explanations from datasets
stages="1 2"
for stage_no in $stages; do
  taskset -c $cpus python source/tagexplainer.py --dataset Mutagenicity --gnn_type $gnn_type --device $device --stage $stage_no
  taskset -c $cpus python source/tagexplainer.py --dataset Proteins --gnn_type $gnn_type --device $device --stage $stage_no
  taskset -c $cpus python source/tagexplainer.py --dataset IMDB-B --gnn_type $gnn_type --device $device --stage $stage_no
  taskset -c $cpus python source/tagexplainer.py --dataset AIDS --gnn_type $gnn_type --device $device --stage $stage_no
  taskset -c $cpus python source/tagexplainer.py --dataset Mutag --gnn_type $gnn_type --device $device --stage $stage_no
  taskset -c $cpus python source/tagexplainer.py --dataset NCI1 --gnn_type $gnn_type --device $device --stage $stage_no
  taskset -c $cpus python source/tagexplainer.py --dataset Graph-SST2 --gnn_type $gnn_type --device $device --stage $stage_no
  taskset -c $cpus python source/tagexplainer.py --dataset REDDIT-B --gnn_type $gnn_type --device cpu --stage $stage_no
  taskset -c $cpus python source/tagexplainer.py --dataset DD --gnn_type $gnn_type --device cpu --stage $stage_no
done


# collect explanations from noisy datasets
stages="1 2"
for stage_no in $stages; do
  taskset -c $cpus python source/tagexplainer.py --dataset Mutagenicity --gnn_type $gnn_type --device $device --stage $stage_no --robustness
  taskset -c $cpus python source/tagexplainer.py --dataset Proteins --gnn_type $gnn_type --device $device --stage $stage_no --robustness
  taskset -c $cpus python source/tagexplainer.py --dataset IMDB-B --gnn_type $gnn_type --device $device --stage $stage_no --robustness
  taskset -c $cpus python source/tagexplainer.py --dataset AIDS --gnn_type $gnn_type --device $device --stage $stage_no --robustness
  wait
done

# stability seeds
stages="1 2"
seeds="2 3"
for stage_no in $stages; do
  for seed in $seeds; do
    taskset -c $cpus python source/tagexplainer.py --dataset Mutagenicity --gnn_type $gnn_type --device $device --stage $stage_no --explainer_run $seed
    taskset -c $cpus python source/tagexplainer.py --dataset Proteins --gnn_type $gnn_type --device $device --stage $stage_no --explainer_run $seed
    taskset -c $cpus python source/tagexplainer.py --dataset IMDB-B --gnn_type $gnn_type --device $device --stage $stage_no --explainer_run $seed
    taskset -c $cpus python source/tagexplainer.py --dataset AIDS --gnn_type $gnn_type --device $device --stage $stage_no --explainer_run $seed
    taskset -c $cpus python source/tagexplainer.py --dataset Mutag --gnn_type $gnn_type --device $device --stage $stage_no --explainer_run $seed
    taskset -c $cpus python source/tagexplainer.py --dataset NCI1 --gnn_type $gnn_type --device $device --stage $stage_no --explainer_run $seed
  done
done

# stability base
stages="1 2"
gnn_types="gat gin sage"
for stage_no in $stages; do
  for gnn_type in $gnn_types; do
    taskset -c $cpus python source/tagexplainer.py --dataset Mutagenicity --gnn_type $gnn_type --device $device --stage $stage_no
    taskset -c $cpus python source/tagexplainer.py --dataset Proteins --gnn_type $gnn_type --device $device --stage $stage_no
    taskset -c $cpus python source/tagexplainer.py --dataset IMDB-B --gnn_type $gnn_type --device $device --stage $stage_no
    taskset -c $cpus python source/tagexplainer.py --dataset AIDS --gnn_type $gnn_type --device $device --stage $stage_no
    taskset -c $cpus python source/tagexplainer.py --dataset Mutag --gnn_type $gnn_type --device $device --stage $stage_no
    taskset -c $cpus python source/tagexplainer.py --dataset NCI1 --gnn_type $gnn_type --device $device --stage $stage_no
  done
done

## generate results
gnn_type=gcn
explainers="tagexplainer_1 tagexplainer_2"
metrics="faithfulness faithfulness_with_removal stability_noise stability_seed stability_base"
for metric in $metrics; do
  for explainer in $explainers; do
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutagenicity --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Proteins --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset IMDB-B --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset AIDS --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
    if ! (("$metric" = "stability_noise")); then
      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutag --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset NCI1 --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
    fi
    if [ "$metric" = "faithfulness" ]; then
      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Graph-SST2 --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric"
      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset REDDIT-B --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric"
      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset DD --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric"
    fi
  done
done
