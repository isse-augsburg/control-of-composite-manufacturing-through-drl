from pathlib import Path
from RlClient.src.evaluation import Evaluator, Baseline, plot_comparison, plot_model
from RlClient.src.training import reward_fn_dryspot_ff_uniformity, reward_fn_weighted_mse, reward_fn_backward_mse, reward_fn_v3
from stable_baselines3 import A2C, PPO

if __name__ == "__main__":
    reward_fn = reward_fn_v3
    server = "http://localhost:8080"

    nenvs = 4

    # model = Baseline("discrete", 3, nenvs, 5) # if you want to see the baseline (constant max. pressure on all inlets)

    model_path = Path("") # TODO path to the <model>.zip 
    model = PPO.load(model_path, print_system_info=True)
    
    e = Evaluator(
        server, 
        reward_fn, 
        nenvs=nenvs, 
        inlets=3, 
        action_type="discrete", 
        num_discrete_actions=5,
        use_fvc=True,
        use_pressure=False
    )

    filelist = [f"mesh{i}.jld2" for i in range(1, 101)] # create the list of filenames that will be evaluated. Path is taken from config.csv
    save_path = Path("") # TODO your save path

    e.eval_filelist(model, filelist=filelist, save_path=save_path)


