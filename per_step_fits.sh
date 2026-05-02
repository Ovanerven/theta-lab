# mof4
python last-layer-ode/analysis/per_step_theta_fit.py --dataset datasets/mof_synthesis_4.npz --scaffold mof_synthesis_4 --sample-idx 0 --gd-steps 400 --n-samples 100 --out results/final_mof_synthesis_4_theta_fit_100samples

# single enzyme
python last-layer-ode/analysis/per_step_theta_fit.py --dataset datasets/single_enzyme_lumped.npz --scaffold single_enzyme_lumped --sample-idx 0 --gd-steps 400 --n-samples 100 --out results/final_single_enzyme_lumped_theta_fit_100samples

# methane 
python last-layer-ode/analysis/per_step_theta_fit.py --dataset datasets/gri30_obs7.npz --scaffold methane_revWGS_ohgate_no --sample-idx 0 --gd-steps 400 --n-samples 100 --out results/final_methane_revWGS_ohgate_no_theta_fit_100samples

# txtl
python last-layer-ode/analysis/per_step_theta_fit.py --dataset datasets/real_ivtt_full.npz --scaffold txtl_resource_and_maturation_dna --sample-idx 0 --gd-steps 400 --loss-species mm,pm --show-species mm,pm --n-samples 100 --out results/final_txtl_maturation_theta_fit_100samples