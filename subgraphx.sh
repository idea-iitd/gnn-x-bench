gnn_type=gcn
device=1
cpus=16-31

# collect explanations from datasets
taskset -c $cpus python source/subgraphx.py --dataset Mutagenicity --gnn_type $gnn_type --device $device --explain_test_only &
taskset -c $cpus python source/subgraphx.py --dataset Mutag --gnn_type $gnn_type --device $device --explain_test_only &
taskset -c $cpus python source/subgraphx.py --dataset IMDB-B --gnn_type $gnn_type --device $device --explain_test_only &
taskset -c $cpus python source/subgraphx.py --dataset Proteins --gnn_type $gnn_type --device $device --explain_test_only &
taskset -c $cpus python source/subgraphx.py --dataset AIDS --gnn_type $gnn_type --device $device --explain_test_only &
taskset -c $cpus python source/subgraphx.py --dataset NCI1 --gnn_type $gnn_type --device $device --explain_test_only &

# generate results
#gnn_type=gcn
#explainers="subgraphx"
#metrics="faithfulness faithfulness_with_removal"
#for metric in $metrics; do
#  for explainer in $explainers; do
#    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutagenicity --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
#    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Proteins --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
#    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset IMDB-B --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
#    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset AIDS --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
#    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset Mutag --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
#    taskset -c $cpus python source/result_generator.py --explainer_name "$explainer" --dataset NCI1 --gnn_type "$gnn_type" --device $device --explanation_metric "$metric"
#  done
#done
