device=1
gnn_type=gcn
cpus=16-31

# reproducibility_reverse for PGExplainer
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name pgexplainer --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset Proteins --gnn_type $gnn_type --explainer_name pgexplainer --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name pgexplainer --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset AIDS --gnn_type $gnn_type --explainer_name pgexplainer --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutag --gnn_type $gnn_type --explainer_name pgexplainer --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset NCI1 --gnn_type $gnn_type --explainer_name pgexplainer --device $device

# reproducibility_reverse for TAGExplainer_1
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name tagexplainer_1 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset Proteins --gnn_type $gnn_type --explainer_name tagexplainer_1 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name tagexplainer_1 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset AIDS --gnn_type $gnn_type --explainer_name tagexplainer_1 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutag --gnn_type $gnn_type --explainer_name tagexplainer_1 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset NCI1 --gnn_type $gnn_type --explainer_name tagexplainer_1 --device $device

# reproducibility_reverse for RCExplainer
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name rcexplainer_1.0 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset Proteins --gnn_type $gnn_type --explainer_name rcexplainer_1.0 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name rcexplainer_1.0 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset AIDS --gnn_type $gnn_type --explainer_name rcexplainer_1.0 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutag --gnn_type $gnn_type --explainer_name rcexplainer_1.0 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset NCI1 --gnn_type $gnn_type --explainer_name rcexplainer_1.0 --device $device

# reproducibility_reverse for GNNExplainer
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name gnnexplainer --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset Proteins --gnn_type $gnn_type --explainer_name gnnexplainer --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name gnnexplainer --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset AIDS --gnn_type $gnn_type --explainer_name gnnexplainer --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutag --gnn_type $gnn_type --explainer_name gnnexplainer --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset NCI1 --gnn_type $gnn_type --explainer_name gnnexplainer --device $device

# reproducibility_reverse for CFF
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name cff_1.0 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset Proteins --gnn_type $gnn_type --explainer_name cff_1.0 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name cff_1.0 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset AIDS --gnn_type $gnn_type --explainer_name cff_1.0 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutag --gnn_type $gnn_type --explainer_name cff_1.0 --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset NCI1 --gnn_type $gnn_type --explainer_name cff_1.0 --device $device

# reproducibility_reverse for GEM
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name gem --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset Proteins --gnn_type $gnn_type --explainer_name gem --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name gem --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset AIDS --gnn_type $gnn_type --explainer_name gem --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutag --gnn_type $gnn_type --explainer_name gem --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset NCI1 --gnn_type $gnn_type --explainer_name gem --device $device

# reproducibility_reverse for SubgraphX
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name subgraphx --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name subgraphx --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset AIDS --gnn_type $gnn_type --explainer_name subgraphx --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset Mutag --gnn_type $gnn_type --explainer_name subgraphx --device $device
taskset -c $cpus python source/reproducibility_reverse.py --dataset NCI1 --gnn_type $gnn_type --explainer_name subgraphx --device $device
