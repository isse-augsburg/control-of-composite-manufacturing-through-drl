import grequests
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvObs
from collections import OrderedDict
from typing import Any, List, Tuple, Union
from gym import spaces, Env
from pathlib import Path
import json
from PIL import Image
from my_utils import detect_dryspots
import datetime

import numpy as np


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
            A list or tuple of observations, one per environment.
            Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(
        obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(
            space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(
            obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, spaces.Tuple):
        assert isinstance(
            obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)


class RemoteRtmEnv(VecEnv, Env):

    def __init__(
        self,
        server_list: list,
        envs_per_server: int,
        reward_fn,
        mode,
        shared_directory:Path,
        img_shape=(50, 50),
        use_fvc=True,
        use_pressure=True,
        action_type="box",
        inlets = 3,
        num_discrete_actions = 11.

    ) -> None:

        self.envs_per_server = envs_per_server
        self.num_servers = len(server_list)
        self.num_envs = self.num_servers * self.envs_per_server
        self.server_list = server_list
        self.server_ids = []
        self.render_mode = None

        # spaces
        # create observation space dependent on if fvc map is layer of image
        self.use_fvc = use_fvc
        self.use_pressure = use_pressure
        if self.use_fvc and self.use_pressure:
            channels = 3
        elif self.use_fvc or self.use_pressure:
            channels = 2
        else:
            channels = 1
        
        self.shape = (channels, img_shape[0], img_shape[1])
        self.inlets = inlets

        assert action_type == "box" or action_type == "discrete", "Invalid action type, must be 'box' or 'discrete'"
        self.action_type = action_type

        if self.action_type == "box":
            self.action_space = spaces.Box(np.full(self.inlets, -1.0, dtype=np.float64), np.full(
                self.inlets, 1.0, dtype=np.float64), dtype=np.float64)
        elif self.action_type == "discrete":
            self.num_discrete_actions = num_discrete_actions
            space = [self.num_discrete_actions for _ in range(self.inlets)]
            self.action_space = spaces.MultiDiscrete(space) 
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8)

        self.actions = None

        self.storage = shared_directory
        self.reward_fn = reward_fn
        self.count = 0
        self.fvc_imgs = [None for _ in range(self.num_envs)]

        # set up servers
        assert mode == "training" or mode == "eval", "mode has to be 'training' or 'eval'."
        
        rs = (grequests.get(f"{url}/setup", json=mode)
              for i, url in enumerate(self.server_list))
        print("Registered server ids:")
        responses = grequests.map(rs)
        for r in responses:
            print(r)
            j = json.loads(r.content)
            id = j["id"]
            print(id)
            self.server_ids.append(id)

    def reset(self):
        print("Reset all.")
        # http request
        rs = (grequests.get(f"{url}/reset") for url in self.server_list)
        responses = grequests.map(rs)

        # create empty images
        imgs = [np.zeros((self.shape[1], self.shape[2]), dtype=np.uint8)
            for _ in range(self.num_envs)]

        pressure_img = np.zeros((self.shape[1], self.shape[2]), dtype=np.uint8)

        # use flowfront, pressure, fvc
        if self.use_fvc and self.use_pressure:
            # load all fvc maps and pressure images and stack them with the flowfront images
            for i, addr in enumerate(self.server_ids):
                for id in range(1, self.envs_per_server + 1):
                    global_id = i * self.envs_per_server + id - 1
                    f = "server" + addr + "env" + str(id) + "fvc.png"
                    with Image.open(self.storage / f) as _img:
                            fvc_img = np.asarray(_img)
                    self.fvc_imgs[global_id] = fvc_img

                    imgs[global_id] = np.stack([imgs[global_id], fvc_img, pressure_img], axis=0)

        # use flowfront, fvc
        elif self.use_fvc:
            # load all fvc maps and stack them with the flowfront images
            for i, addr in enumerate(self.server_ids):
                for id in range(1, self.envs_per_server + 1):
                    global_id = i * self.envs_per_server + id - 1
                    f = "server" + addr + "env" + str(id) + "fvc.png"
                    with Image.open(self.storage / f) as _img:
                            fvc_img = np.asarray(_img)
                    self.fvc_imgs[global_id] = fvc_img

                    imgs[global_id] = np.stack([imgs[global_id], fvc_img], axis=0)

        # use flowfront, pressure
        elif self.use_pressure:
            # load all fvc maps and stack them with the flowfront images  
            for i, addr in enumerate(self.server_ids):
                for id in range(1, self.envs_per_server + 1):
                    global_id = i * self.envs_per_server + id - 1

                    imgs[global_id] = np.stack([imgs[global_id], pressure_img], axis=0)          
        
        # use flowfront
        else:
            for i, img in enumerate(imgs):
                imgs[i] = np.expand_dims(img, 0)
                    
        imgs = _flatten_obs(imgs, self.observation_space)
        return imgs

    def select_file(self, tuples:list):
        assert self.num_servers == 1, "Manual selection is only allowed with 1 server in use."
        t = []
        
        for id, path in tuples:
            t.append((id, path))
        
        url = self.server_list[0]
        r = grequests.get(f"{url}/file_select", json=json.dumps(t))
        responses = grequests.map([r])

        imgs = [np.zeros((self.shape[1], self.shape[2]), dtype=np.uint8)
                for _ in range(len(tuples))]

        pressure_img = np.zeros((self.shape[1], self.shape[2]), dtype=np.uint8)

        # use flowfront, fvc, pressure
        if self.use_fvc and self.use_pressure:
             # load all fvc maps and stack them with the images
            addr = self.server_ids[0]
            for i, (id, _) in enumerate(tuples):
                # special case for global_id, bc only one server is in use
                global_id = id - 1
                f = "server" + addr + "env" + str(id) + "fvc.png"
                with Image.open(self.storage / f) as _img:
                        fvc_img = np.asarray(_img)
                self.fvc_imgs[global_id] = fvc_img

                imgs[i] = np.stack([imgs[i], fvc_img, pressure_img], axis=0)

        # use flowfront, fvc
        elif self.use_fvc:
            # load all fvc maps and stack them with the images
            addr = self.server_ids[0]
            for i, (id, _) in enumerate(tuples):
                # special case for global_id, bc only one server is in use
                global_id = id - 1
                f = "server" + addr + "env" + str(id) + "fvc.png"
                with Image.open(self.storage / f) as _img:
                        img = np.asarray(_img)
                self.fvc_imgs[global_id] = img
                imgs[i] = np.stack([imgs[i], img], axis=0)

        # use flowfront, pressure
        elif self.use_pressure:
            # load all fvc maps and stack them with the flowfront images
            addr = self.server_ids[0] 
            for i, (id, _) in enumerate(tuples):
                # special case for global_id, bc only one server is in use
                global_id = id - 1
                imgs[i] = np.stack([imgs[i], pressure_img], axis=0)

        # use flowfront
        else: 
            for i, img in enumerate(imgs):
                imgs[i] = np.expand_dims(img, 0)

        imgs = _flatten_obs(imgs, self.observation_space)
        return imgs

    def step(self, actions): 
        ct = datetime.datetime.now()
        self.count += 1
        # print(f"Step {self.count}: {ct}")

        if self.action_type == "box":
            actions = (actions + 1.) / 2.
        elif self.action_type == "discrete":
            actions = actions / (self.num_discrete_actions - 1)
        
        conditions = []
        for id in range(self.num_servers):
            server_actions = actions[id * self.envs_per_server: (id + 1) * self.envs_per_server]
            conditions.append([action.tolist() for action in server_actions])
        
        # http request
        rs = (grequests.get(
            f"{url}/step", json=json.dumps(conditions[i])) for i, url in enumerate(self.server_list))
        responses = grequests.map(rs)

        # read responses
        sim_timeout, ts, addrs = self.responses(responses)

        # get & analyze images per server, calculate rewards
        imgs, finisheds, rewards, infos = self.observe(ts, sim_timeout, addrs)

        # reset environments that are either filled or have a dryspot
        # create a list of {0, 1} for each server, that flags for every environment if it shall be resetted or not
        selections = []
        for id in range(self.num_servers):
            selection = finisheds[id *
                                  self.envs_per_server: (id + 1) * self.envs_per_server]
            selection = [int(x) for x in selection]
            selections.append(selection)

        # http request for selective reset
        # note that grequests.map keeps the order of requests/ responses, which is important
        rs = (grequests.get(f"{url}/reset_selected", json=json.dumps(
            selections[i])) for i, url in enumerate(self.server_list))
        responses = grequests.map(rs)

        # optionally set fvc image to resetted observations
        if self.use_fvc:
            for i, sel in enumerate(selections):
                for id in range(1, self.envs_per_server + 1):
                    global_id = i * self.envs_per_server + id - 1
                    if sel[id - 1] == 1: # test if the flag is set to 1
                        addr = self.server_ids[i]
                        f = "server" + addr + "env" + str(id) + "fvc.png"
                        with Image.open(self.storage / f) as _img:
                            fvc_img = np.asarray(_img)

                        self.fvc_imgs[global_id] = fvc_img
                        imgs[global_id][1, :, :] = fvc_img # if fvc is used, it is definitely in dimension 1

        imgs = _flatten_obs(imgs, self.observation_space)
        return imgs, rewards, finisheds, infos

    def fvc_maps(self):
        rs = (grequests.get(f"{url}/fvc_map") for url in self.server_list)
        responses = grequests.map(rs)

        imgs = []
        for i, addr in enumerate(self.server_ids):
            for j in range(1, self.envs_per_server + 1):
                f = "server" + addr + "env" + str(j) + "fvc.png"
                img = Image.open(self.storage / f)
                imgs.append(np.asarray(img))
        return imgs

    def observe(self, ts, sim_timeouts, addrs):
        imgs = []
        finisheds = []
        rewards = []
        infos = []

        # collect and process images etc. for every server and all of it's environments
        for i in range(self.num_servers):
            for id in range(1, self.envs_per_server + 1):
                global_id = i * self.envs_per_server + id - 1
                fn = "server" + addrs[i] + "env" + str(id) + "image.png"
                fn_lines = "server" + addrs[i] + "env" + str(id) + "image_lines.png"

                with Image.open(self.storage / fn) as _img:
                    ff_img = np.asarray(_img)
                
                filled = np.mean(ff_img) > 0.99 * 255.
                # print(f"Filled: {np.mean(img) / 255.:.2f}")
                dryspot = detect_dryspots(ff_img)

                finished = sim_timeouts[global_id] or filled or dryspot

                # call the reward function
                # !IMPORTANT! call reward function before the image potentially gets overwritten because of a reset of the environement,
                # so the reward gets calculated for the terminal image
                reward = self.reward_fn(ff_img, ts[global_id], dryspot, filled, sim_timeouts[global_id], self.storage / fn_lines) 

                # Optionally stack pressure and fvc image, else add 'empty' channel dim
                if self.use_fvc and self.use_pressure:
                    f = "server" + addrs[i] + "env" + str(id) + "pressure.png"
                    with Image.open(self.storage / f) as _img:
                        pressure_img = np.asarray(_img)
                    fvc = self.fvc_imgs[global_id]
                    img = np.stack([ff_img, fvc, pressure_img], axis=0)

                elif self.use_fvc:
                    fvc = self.fvc_imgs[global_id]
                    img = np.stack([ff_img, fvc], axis=0)

                elif self.use_pressure:
                    f = "server" + addrs[i] + "env" + str(id) + "pressure.png"
                    with Image.open(self.storage / f) as _img:
                        pressure_img = np.asarray(_img)
                    img = np.stack([ff_img, pressure_img], axis=0) 

                else: 
                    img = np.expand_dims(ff_img, 0)

                if finished:
                    # print(f"Finished, filling: {np.mean(img) / 255.:.2f}")
                    info = {"terminal_observation": img}

                    if self.use_fvc and self.use_pressure:
                        # create empty 2 channel image, fvc channel must be overwritten after the server env was resetted
                        img = np.zeros((3, self.shape[1], self.shape[2]), dtype=np.uint8)
                    elif self.use_fvc or self.use_pressure:
                        # create empty 2 channel image, fvc channel must be overwritten after the server env was resetted
                        img = np.zeros((2, self.shape[1], self.shape[2]), dtype=np.uint8)
                    else: 
                        img = np.zeros((1, self.shape[1], self.shape[2]), dtype=np.uint8)
                else:
                    info = {}

                imgs.append(img)
                finisheds.append(finished)
                rewards.append(reward)
                infos.append(info)

        finisheds = np.array(finisheds)
        rewards = np.array(rewards)
        infos = tuple(i for i in infos)
        return imgs, finisheds, rewards, infos

    def responses(self, rs):
        sim_timeouts = []
        ts = []
        addrs = []
        for r in rs:
            j = json.loads(r.content)
            t = j["t"]
            finished = j["finished"]
            addr = j["addr"]
            ts.extend(t)
            sim_timeouts.extend(finished)
            addrs.append(addr)

        sim_timeouts = np.array(sim_timeouts)
        ts = np.array(ts)
        return sim_timeouts, ts, addrs

    def close(self) -> None:
        rs = (grequests.get(f"{url}/close") for url in self.server_list)
        grequests.map(rs)
        return None

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in range(self.num_envs)]

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        return None

    def get_attr(self, attr_name: str, indices=None):
        if (attr_name == "render_mode"):
            return [None for _ in range(self.num_envs)]
        return None

    def seed(self, seed=None):
        return None

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        return None

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        return self.step(self.actions)

