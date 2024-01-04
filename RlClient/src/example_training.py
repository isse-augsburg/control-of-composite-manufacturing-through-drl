from pathlib import Path
import time
from remote_rtm_env import RemoteRtmEnv
from training import reward_fn_backward_mse, train_model, reward_fn_weighted_mse, reward_fn_backward_mse, reward_fn_v3
from stable_baselines3 import A2C, PPO
from custom_policy import Rl4RtmActorCriticPolicy, WeNeedToGoDeeper
from evaluation import Evaluator



if __name__ == "__main__":

    # TODO List your servers here.
    # The last one will be used for evaluation during training
    servers = [
        "http://172.17.0.2:8080",
        "http://172.17.0.3:8080"
    ]
    envs_per_server = 2

    base_path = Path("/cfs/results/example_training") # TODO path where you want to save checkpoints etc.
    steps = 32 # TODO adjust training steps

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
        tensorboard_log=None, 
        n_steps=4, # TODO rollout steps per policy update
        seed=seed, 
        batch_size=2, # TODO batchsize
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
        log_interval=8, # TODO
        eval_freq=8, # TODO
        save_freq=16, # TODO
        use_fvc=True,
        use_pressure=True,
        action_type=action_type,
        num_discrete_actions=num_discrete_actions
    )
    total_time = time.time() - start_time
    print(f"Finished. {total_time}")
    with open(log_path / "timing.log", "w") as f:
        f.write(f"Steps: {steps}\n")
        f.write(f"Envs: {envs_per_server * (len(servers) - 1)} (distributed on {len(servers) - 1} servers)\n")
        f.write(f"Time: {total_time}s\n")
