gnn_type=gcn
device=3
cpus=16-31

# generate results for GNNExplainer, PGExplainer, TAGExplainer, CFF, RCExplainer
gnn_type=gcn
explainers="gnnexplainer pgexplainer tagexplainer_1 tagexplainer_2 cff_1.0 rcexplainer_1.0"
metrics="faithfulness faithfulness_with_removal stability_noise stability_seed stability_base"
for metric in $metrics; do
  for explainer in $explainers; do
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutagenicity --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Proteins --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset IMDB-B --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset AIDS --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    if ! (("$metric" = "stability_noise")); then
      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutag --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset NCI1 --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    fi
    if [ "$metric" = "faithfulness" ]; then
      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Graph-SST2 --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric" --folded
      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset REDDIT-B --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric" --folded
      taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset DD --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric" --folded
    fi
  done
done

# generate results for GEM, SubgraphX
gnn_type=gcn
explainers="gem subgraphx"
metrics="faithfulness faithfulness_with_removal"
for metric in $metrics; do
  for explainer in $explainers; do
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutagenicity --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Proteins --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset IMDB-B --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset AIDS --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutag --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset NCI1 --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
  done
done

# generate results for inductive methods
gnn_type=gcn
explainers="pgexplainer rcexplainer_1.0 tagexplainer_1 tagexplainer_2 gem"
metrics="faithfulness_on_test"
for metric in $metrics; do
  for explainer in $explainers; do
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutagenicity --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Proteins --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset IMDB-B --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset AIDS --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutag --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset NCI1 --gnn_type "$gnn_type" --device $device --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Graph-SST2 --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset DD --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric" --folded
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset REDDIT-B --gnn_type "$gnn_type" --device cpu --explanation_metric "$metric" --folded
  done
done