#RCExplainer
python source/cf_result_generator_avg.py --dataset AIDS --explainer_name rcexplainer_0.0 \
--explanation_metric stability_base stability_seed stability_noise feasibility size sparsity sufficiency 1> cf_results/rc_aids.txt
python source/cf_result_generator_avg.py --dataset Proteins --explainer_name rcexplainer_0.0 \
--explanation_metric stability_base stability_seed stability_noise stability_feature_noise stability_adversarial_noise feasibility size sparsity sufficiency 1> cf_results/rc_proteins.txt
python source/cf_result_generator_avg.py --dataset Mutagenicity --explainer_name rcexplainer_0.0 \
--explanation_metric stability_base stability_seed stability_noise stability_feature_noise feasibility size sparsity sufficiency 1> cf_results/rc_mutagenicity.txt
python source/cf_result_generator_avg.py --dataset IMDB-B --explainer_name rcexplainer_0.0 \
--explanation_metric stability_base stability_seed stability_noise feasibility size sparsity sufficiency 1> cf_results/rc_imdb.txt
python source/cf_result_generator_avg.py --dataset Mutag --explainer_name rcexplainer_0.0 \
--explanation_metric stability_base stability_seed stability_feature_noise feasibility size sparsity sufficiency 1> cf_results/rc_mutag.txt
#CF^2
# python source/cf_result_generator_avg.py --dataset AIDS --explainer_name cff_0.0 \
# --explanation_metric stability_base stability_seed stability_noise feasibility size sparsity sufficiency > cf_results/cff_aids.txt
# python source/cf_result_generator_avg.py --dataset Proteins --explainer_name cff_0.0 \
# --explanation_metric stability_base stability_seed stability_noise feasibility size sparsity sufficiency > cf_results/cff_proteins.txt
# python source/cf_result_generator_avg.py --dataset Mutagenicity --explainer_name cff_0.0 \
# --explanation_metric stability_base stability_seed stability_noise feasibility size sparsity sufficiency > cf_results/cff_mutagenicity.txt
# python source/cf_result_generator_avg.py --dataset IMDB-B --explainer_name cff_0.0 \
# --explanation_metric stability_base stability_seed stability_noise feasibility size sparsity sufficiency > cf_results/cff_imdb.txt
# python source/cf_result_generator_avg.py --dataset Mutag --explainer_name cff_0.0 \
# --explanation_metric stability_base stability_seed feasibility size sparsity sufficiency > cf_results/cff_mutag.txt

# #CLEAR
# python source/cf_result_generator_avg.py --dataset AIDS --explainer_name clear \
# --explanation_metric stability_base stability_seed stability_noise feasibility size sparsity sufficiency > cf_results/clear_aids.txt
# python source/cf_result_generator_avg.py --dataset Proteins --explainer_name clear \
# --explanation_metric stability_base stability_seed stability_noise feasibility size sparsity sufficiency > cf_results/clear_proteins.txt
# python source/cf_result_generator_avg.py --dataset Mutagenicity --explainer_name clear \
# --explanation_metric stability_base stability_seed stability_noise feasibility size sparsity sufficiency > cf_results/clear_mutagenicity.txt
# python source/cf_result_generator_avg.py --dataset IMDB-B --explainer_name clear \
# --explanation_metric stability_base stability_seed stability_noise feasibility size sparsity sufficiency > cf_results/clear_imdb.txt
# python source/cf_result_generator_avg.py --dataset Mutag --explainer_name clear \
# --explanation_metric stability_base stability_seed feasibility size sparsity sufficiency > cf_results/clear_mutag.txt