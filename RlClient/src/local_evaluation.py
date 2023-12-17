from pathlib import Path
from RlClient.src.evaluation import Evaluator, Baseline, plot_comparison, plot_model
from RlClient.src.training import reward_fn_dryspot_ff_uniformity, reward_fn_weighted_mse, reward_fn_backward_mse, reward_fn_v3
from stable_baselines3 import A2C, PPO

if __name__ == "__main__":
    reward_fn = reward_fn_v3
    server = "http://localhost:8080"

    nenvs = 16

    # model = Baseline("discrete", 3, nenvs, 5)
    model = PPO.load(r"X:\h\e\heberleo\RL4RTM_paper\v1\slight\training\FfP\PPO\best_model.zip", print_system_info=True)
    
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

    filelist = [f"mesh{i}.jld2" for i in range(1, 101)]
    save_path = Path(r"X:\h\e\heberleo\RL4RTM_paper\v1\slight\statistical_evaluation\FfP\PPO")

    e.eval_filelist(model, filelist=filelist, save_path=save_path)


