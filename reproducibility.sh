device=1
gnn_type=gcn
cpus=16-31

# reproducibility for PGExplainer
taskset -c $cpus python source/reproducibility.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name pgexplainer --device $device
taskset -c $cpus python source/reproducibility.py --dataset Proteins --gnn_type $gnn_type --explainer_name pgexplainer --device $device
taskset -c $cpus python source/reproducibility.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name pgexplainer --device $device
taskset -c $cpus python source/reproducibility.py --dataset AIDS --gnn_type $gnn_type --explainer_name pgexplainer --device $device
taskset -c $cpus python source/reproducibility.py --dataset Mutag --gnn_type $gnn_type --explainer_name pgexplainer --device $device
taskset -c $cpus python source/reproducibility.py --dataset NCI1 --gnn_type $gnn_type --explainer_name pgexplainer --device $device

# reproducibility for TAGExplainer_1
taskset -c $cpus python source/reproducibility.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name tagexplainer_1 --device $device
taskset -c $cpus python source/reproducibility.py --dataset Proteins --gnn_type $gnn_type --explainer_name tagexplainer_1 --device $device
taskset -c $cpus python source/reproducibility.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name tagexplainer_1 --device $device
taskset -c $cpus python source/reproducibility.py --dataset AIDS --gnn_type $gnn_type --explainer_name tagexplainer_1 --device $device
taskset -c $cpus python source/reproducibility.py --dataset Mutag --gnn_type $gnn_type --explainer_name tagexplainer_1 --device $device
taskset -c $cpus python source/reproducibility.py --dataset NCI1 --gnn_type $gnn_type --explainer_name tagexplainer_1 --device $device

# reproducibility for RCExplainer
taskset -c $cpus python source/reproducibility.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name rcexplainer_1.0 --device $device
taskset -c $cpus python source/reproducibility.py --dataset Proteins --gnn_type $gnn_type --explainer_name rcexplainer_1.0 --device $device
taskset -c $cpus python source/reproducibility.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name rcexplainer_1.0 --device $device
taskset -c $cpus python source/reproducibility.py --dataset AIDS --gnn_type $gnn_type --explainer_name rcexplainer_1.0 --device $device
taskset -c $cpus python source/reproducibility.py --dataset Mutag --gnn_type $gnn_type --explainer_name rcexplainer_1.0 --device $device
taskset -c $cpus python source/reproducibility.py --dataset NCI1 --gnn_type $gnn_type --explainer_name rcexplainer_1.0 --device $device

# reproducibility for GNNExplainer
taskset -c $cpus python source/reproducibility.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name gnnexplainer --device $device
taskset -c $cpus python source/reproducibility.py --dataset Proteins --gnn_type $gnn_type --explainer_name gnnexplainer --device $device
taskset -c $cpus python source/reproducibility.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name gnnexplainer --device $device
taskset -c $cpus python source/reproducibility.py --dataset AIDS --gnn_type $gnn_type --explainer_name gnnexplainer --device $device
taskset -c $cpus python source/reproducibility.py --dataset Mutag --gnn_type $gnn_type --explainer_name gnnexplainer --device $device
taskset -c $cpus python source/reproducibility.py --dataset NCI1 --gnn_type $gnn_type --explainer_name gnnexplainer --device $device

# reproducibility for CFF
taskset -c $cpus python source/reproducibility.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name cff_1.0 --device $device
taskset -c $cpus python source/reproducibility.py --dataset Proteins --gnn_type $gnn_type --explainer_name cff_1.0 --device $device
taskset -c $cpus python source/reproducibility.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name cff_1.0 --device $device
taskset -c $cpus python source/reproducibility.py --dataset AIDS --gnn_type $gnn_type --explainer_name cff_1.0 --device $device
taskset -c $cpus python source/reproducibility.py --dataset Mutag --gnn_type $gnn_type --explainer_name cff_1.0 --device $device
taskset -c $cpus python source/reproducibility.py --dataset NCI1 --gnn_type $gnn_type --explainer_name cff_1.0 --device $device

# reproducibility for GEM
taskset -c $cpus python source/reproducibility.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name gem --device $device
taskset -c $cpus python source/reproducibility.py --dataset Proteins --gnn_type $gnn_type --explainer_name gem --device $device
taskset -c $cpus python source/reproducibility.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name gem --device $device
taskset -c $cpus python source/reproducibility.py --dataset AIDS --gnn_type $gnn_type --explainer_name gem --device $device
taskset -c $cpus python source/reproducibility.py --dataset Mutag --gnn_type $gnn_type --explainer_name gem --device $device
taskset -c $cpus python source/reproducibility.py --dataset NCI1 --gnn_type $gnn_type --explainer_name gem --device $device

# reproducibility for SubgraphX
taskset -c $cpus python source/reproducibility.py --dataset Mutagenicity --gnn_type $gnn_type --explainer_name subgraphx --device $device
taskset -c $cpus python source/reproducibility.py --dataset Proteins --gnn_type $gnn_type --explainer_name subgraphx --device $device
taskset -c $cpus python source/reproducibility.py --dataset IMDB-B --gnn_type $gnn_type --explainer_name subgraphx --device $device
taskset -c $cpus python source/reproducibility.py --dataset AIDS --gnn_type $gnn_type --explainer_name subgraphx --device $device
taskset -c $cpus python source/reproducibility.py --dataset Mutag --gnn_type $gnn_type --explainer_name subgraphx --device $device
taskset -c $cpus python source/reproducibility.py --dataset NCI1 --gnn_type $gnn_type --explainer_name subgraphx --device $device
