gnn_type=gcn
device=0
cpus=0-15

# generate ground truth explanations
taskset -c $cpus python source/gem_gt.py --dataset Mutagenicity --gnn_type $gnn_type --device $device
taskset -c $cpus python source/gem_gt.py --dataset Proteins --gnn_type $gnn_type --device $device
taskset -c $cpus python source/gem_gt.py --dataset IMDB-B --gnn_type $gnn_type --device $device
taskset -c $cpus python source/gem_gt.py --dataset Mutag --gnn_type $gnn_type --device $device
taskset -c $cpus python source/gem_gt.py --dataset AIDS --gnn_type $gnn_type --device $device
taskset -c $cpus python source/gem_gt.py --dataset NCI1 --gnn_type $gnn_type --device $device

# collect explanations from datasets
taskset -c $cpus python source/gem.py --dataset Mutagenicity --gnn_type $gnn_type --device $device --weighted --gae3 --early_stop --normalize_feat --train_on_positive_label
taskset -c $cpus python source/gem.py --dataset Proteins --gnn_type $gnn_type --device $device --weighted --gae3 --early_stop --normalize_feat --train_on_positive_label
taskset -c $cpus python source/gem.py --dataset IMDB-B --gnn_type $gnn_type --device $device --weighted --gae3 --early_stop --normalize_feat --train_on_positive_label
taskset -c $cpus python source/gem.py --dataset Mutag --gnn_type $gnn_type --device $device --weighted --gae3 --early_stop --normalize_feat --train_on_positive_label
taskset -c $cpus python source/gem.py --dataset AIDS --gnn_type $gnn_type --device $device --weighted --gae3 --early_stop --normalize_feat --train_on_positive_label
taskset -c $cpus python source/gem.py --dataset NCI1 --gnn_type $gnn_type --device $device --weighted --gae3 --early_stop --normalize_feat --train_on_positive_label

# generate results
gnn_type=gcn
explainers="gem"
metrics="faithfulness faithfulness_with_removal faithfulness_on_test"
for metric in $metrics; do
  for explainer in $explainers; do
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutagenicity --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Proteins --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset IMDB-B --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset AIDS --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutag --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset NCI1 --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
  done
done
