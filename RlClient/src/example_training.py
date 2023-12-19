from pathlib import Path
import time
from RlClient.src.remote_rtm_env import RemoteRtmEnv
from RlClient.src.training import reward_fn_backward_mse, train_model, reward_fn_weighted_mse, reward_fn_backward_mse, reward_fn_v3
from stable_baselines3 import A2C, PPO
from RlClient.src.custom_policy import Rl4RtmActorCriticPolicy, WeNeedToGoDeeper
from RlClient.src.evaluation import Evaluator



if __name__ == "__main__":

    # List your servers here.
    # The last one will be used for evaluation during training
    servers = [
        "http://localhost:8080",
        "http://localhost:8081"
    ]
    envs_per_server = 4

    base_path = Path("") # TODO path where you want to save checkpoints etc.
    steps = 2000000

    seed = 1337
    reward_fn = reward_fn_v3
    inlets = 3
    action_type = "discrete"
    num_discrete_actions = 5

    log_path = base_path / ""

    model_fn = lambda env: PPO(
        Rl4RtmActorCriticPolicy, 
        env=env, 
        verbose=2, 
        tensorboard_log=str(log_path / "tensorboard"), 
        n_steps=20, 
        seed=seed, 
        batch_size=279,
        use_sde=False
        )
    
    start_time = time.time()
    train_model(
        model_fn=model_fn,
        reward_fn=reward_fn,
        training_steps=steps,
        save_path=log_path,
        servers=servers,
        envs_per_server=envs_per_server,
        log_interval=10,
        eval_freq=100,
        save_freq=1200,
        use_fvc=True,
        use_pressure=True,
        action_type=action_type,
        num_discrete_actions=num_discrete_actions
    )
    total_time = time.time() - start_time
    print(f"Finished. {total_time}")
    with open(log_path / "timing.log", "w") as f:
        f.write(f"Steps: {steps}\n")
        f.write(f"Envs: {envs_per_server * len(servers - 1)} (distributed on {len(servers - 1)} servers)\n")
        f.write(f"Time: {total_time}s\n")


    print("Evaluation:")
    
    server = servers[1]
 
    e = Evaluator(
        server, 
        reward_fn, 
        nenvs=31, 
        inlets=inlets, 
        action_type=action_type, 
        num_discrete_actions=num_discrete_actions
    )

    model_path = log_path / "best_model.zip"
    model = PPO.load(model_path)

    filelist = [f"mesh{i}.jld2" for i in range(1, 101)]
    save_path = Path("") # TODO path where you want to save evaluation of the best model

    e.eval_filelist(model, filelist=filelist, save_path=save_path)

    print("Done.")


