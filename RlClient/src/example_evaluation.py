from pathlib import Path
from evaluation import Evaluator, Baseline, plot_comparison, plot_model
from training import reward_fn_dryspot_ff_uniformity, reward_fn_weighted_mse, reward_fn_backward_mse, reward_fn_v3
from stable_baselines3 import A2C, PPO

if __name__ == "__main__":
    reward_fn = reward_fn_v3
    server = "http://172.17.0.3:8080"

    nenvs = 2

    # model = Baseline("discrete", 3, nenvs, 5) # if you want to see the baseline (constant max. pressure on all inlets)

    model_path = Path("/cfs/pretrained_agents/PPO_FfFvcP.zip") # PPO agent from paper
    # model_path = Path("/cfs/pretrained_agents/A2C_FfFvcP.zip") # A2C agent from paper

    model = PPO.load(model_path, print_system_info=True)
    # model = A2C.load(model_path, print_system_info=True)
    
    e = Evaluator(
        server, 
        reward_fn, 
        nenvs=nenvs, 
        inlets=3, 
        action_type="discrete", 
        num_discrete_actions=5,
        use_fvc=True,
        use_pressure=True
    )

    filelist = [f"mesh{i}.jld2" for i in range(1, 101)] # create the list of filenames that will be evaluated. Path is taken from config.csv
    save_path = Path("/cfs/results/example_evaluation") 
    e.eval_filelist(model, filelist=filelist, save_path=save_path)
    print("Done.")

