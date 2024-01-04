from pathlib import Path
import numpy as np
from my_utils import get_leftmost_and_rightmost_edges_of_ff_min_max
from remote_rtm_env import RemoteRtmEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


def train_model(
    model_fn,
    reward_fn,
    training_steps,
    save_path,
    servers,
    envs_per_server=31,
    log_interval=10, 
    eval_freq=400,
    save_freq=1200,
    use_fvc=True,
    use_pressure=True,
    seed=1337,
    action_type="box",
    num_discrete_actions=5
):
    # TRAINING ENVIRONMENT
    monitor_path = save_path / "monitoring"
    monitor_path.mkdir(parents=True, exist_ok=True)

    share_path = Path("/cfs/file_exchange")

    env = VecMonitor(
        venv=RemoteRtmEnv(servers[:-1], envs_per_server, reward_fn, "training", share_path, use_fvc=use_fvc, use_pressure=use_pressure, action_type=action_type, num_discrete_actions=num_discrete_actions),
        filename=str(monitor_path)
    )
    print("Training env created.")
    # MODEL
    (save_path/ "tensorboard").mkdir(parents=True, exist_ok=True)
    model = model_fn(env)

    # CHECKPOINT CALLBACK
    # Save a checkpoint every ~250 000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path / "checkpoint",
        name_prefix="temp_model",
        verbose=2
    )

    # EVAL CALLBACK
    eval_env = VecMonitor(
        venv=RemoteRtmEnv([servers[-1]], envs_per_server, reward_fn, "eval", share_path, use_fvc=use_fvc, use_pressure=use_pressure, action_type=action_type, num_discrete_actions=num_discrete_actions)
    )
    print("Validation env created.")
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=str(save_path),
        log_path=str(save_path), 
        eval_freq=eval_freq,
        deterministic=True, 
        render=False,
        n_eval_episodes=31
    )

    # CALLBACK LIST
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # TRAINING
    model.learn(total_timesteps=training_steps, log_interval=log_interval, callback=callbacks)
    model.save(save_path / "last_model.zip")

    # close environment
    env.close()

def reward_fn_base(img, t, dryspot, filled, timeout, some_path): 
    return - int(dryspot) + int(filled)

def reward_fn_dryspot_ff_uniformity(img, t, dryspot, filled, sim_timeouts, colored_lines_img_path):
    min_left, max_left, min_right, max_right = get_leftmost_and_rightmost_edges_of_ff_min_max(img, img_save_path=colored_lines_img_path)
    spread_left = max_left - min_left
    spread_right = max_right - min_right
    ds = (- int(dryspot)) * 100
    fill = int(filled) * 100
    reward = ds + fill + (5 - spread_left) + (5 - spread_right)
    # print(f"Dryspot: {dryspot}, fill: {fill}, spread_left: {5 - spread_left}, spread_right: {5 - spread_right}, Reward: {reward}")
    return reward

def reward_fn_area_loss(img, t, dryspot, filled, sim_timeouts, colored_lines_img_path):
    min_left, max_left, min_right, max_right = get_leftmost_and_rightmost_edges_of_ff_min_max(img, img_save_path=colored_lines_img_path)

    img_slice = img[0, :, min_left:max_right]
    fill_status = np.mean(img_slice) / 255
    if np.isnan(fill_status):
        fill_status = 0.

    img = img[0, :, :]
    an_img = Image.fromarray(img)
    an_img = an_img.convert("RGB")
    draw = ImageDraw.Draw(an_img) 
    draw.line((max_right, 0, max_right, img.shape[1]), fill=(255, 0, 0), width=1)
    draw.line((min_left, 0, min_left, img.shape[1]), fill=(255, 0, 0), width=1)

    f = np.zeros(25)
    for i in range(25):
        fs = np.mean(img_slice[i * 2:(i + 1) * 2, :]) / 255
        if fs > 0.4:
            fs = 1.
        f[i] = fs
        img[i * 2:(i + 1) * 2, min_left:max_right] = fs * 255


    r = -(1. - fill_status) * (max_right - min_left) * 50

    min_fill = np.min(f)
    if np.isnan(min_fill):
        min_fill = 0.
    r_ = -(1. - min_fill) * pow((max_right - min_left), 2)

    #plt.imshow(img)
    #plt.show()
    plt.imshow(np.array(an_img))
    plt.title(str(r_))
    plt.show()

def reward_fn_weighted_mse(img, t, dryspot, filled, sim_timeouts, colored_lines_img_path):
    ds = (- int(dryspot)) * 1000
    fill = int(filled) * 100

    min_left, max_left, min_right, max_right = get_leftmost_and_rightmost_edges_of_ff_min_max(img, img_save_path=colored_lines_img_path)

    img_slice = img[:, min_left:max_right] / 255
    width = img_slice.shape[1]
    half = int((min_left + max_right) / 2)
    if width <= 1:
        reward = ds + fill
        return reward

    start = -int(width / 2)
    stop = int(width / 2) + width % 2
    weights = np.arange(start=start, stop=stop)
    
    weights[int(weights.shape[0] / 2):] += 1

    mask = np.zeros_like(img_slice)
    mask[:, :int(width / 2)] = 1.
    err = mask - img_slice
    err = err * weights
    err = np.power(err, 2)  
    err = np.sum(err) / (width * 50)
    err = -err * 10
    
    reward = ds + fill + err

    return np.float32(reward)

def reward_fn_backward_mse(img, t, dryspot, filled, sim_timeouts, colored_lines_img_path):
    ds = (- int(dryspot)) * 1000
    fill = int(filled) * 100

    min_left, max_left, min_right, max_right = get_leftmost_and_rightmost_edges_of_ff_min_max(img, img_save_path=None)

    img_slice = img[:, :max_right] / 255
    width = img_slice.shape[1]
    if width <= 1:
        reward = ds + fill
        return reward

    start = width
    stop = 0
    weights = np.arange(start=start, stop=stop, step=-1)
    weights[-2:] = 0

    mask = np.ones_like(img_slice)
    
    err = mask - img_slice
    err = err * weights
    err = np.power(err, 2)  
    err = np.sum(err) / (width * 50)
    err = -err * 10
    
    reward = ds + fill + err

    return np.float32(reward)

def reward_fn_v3(img, t, dryspot, filled, sim_timeouts, colored_lines_img_path):
    ds = (- int(dryspot)) * 100
    fill = int(filled)

    n_steps = t / 0.5
    fill = ((1 / n_steps) * 3000) * fill

    min_left, max_left, min_right, max_right = get_leftmost_and_rightmost_edges_of_ff_min_max(img, img_save_path=None)

    img_slice = img[:, :max_right] / 255
    width = img_slice.shape[1]
    if width <= 1:
        reward = ds + fill
        return reward

    start = width
    stop = 0
    weights = np.arange(start=start, stop=stop, step=-1)
    weights[-2:] = 0

    mask = np.ones_like(img_slice)
    
    err = mask - img_slice
    err = err * weights
    err = np.power(err, 2)  
    err = np.sum(err) / (width * 50)
    err = -err * 10
    
    reward = ds + fill + err

    return np.float32(reward)


def reward_fn_v4(img, t, dryspot, filled, sim_timeouts, colored_lines_img_path):
    ds = (- int(dryspot)) * 100
    fill = int(filled) * 100

    min_left, max_left, min_right, max_right = get_leftmost_and_rightmost_edges_of_ff_min_max(img, img_save_path=None)

    img_slice = img[:, :max_right] / 255
    width = img_slice.shape[1]
    if width <= 1:
        reward = ds + fill
        return reward

    start = width
    stop = 0
    weights = np.arange(start=start, stop=stop, step=-1)
    weights[-2:] = 0

    mask = np.ones_like(img_slice)
    
    err = mask - img_slice
    err = err * weights
    err = np.power(err, 2)  
    err = np.sum(err) / (width * 50)
    err = -err 
    
    reward = ds + fill + err

    return np.float32(reward)
 
        


