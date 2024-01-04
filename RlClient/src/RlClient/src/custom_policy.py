from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym as gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space

from stable_baselines3.common.type_aliases import Schedule


class Rl4RtmActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self, 
        observation_space: gym.spaces.Space, 
        action_space: gym.spaces.Space, 
        lr_schedule: Schedule,
        use_sde=False,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=None
    ):

        net_arch = [128, dict(vf=[32], pi=[32])]
        features_extractor_class = NatureCNN
        activation_fn = th.nn.ReLU
        super().__init__(
            observation_space, 
            action_space, 
            lr_schedule, 
            net_arch, 
            activation_fn, 
            ortho_init=True, 
            use_sde=use_sde, 
            log_std_init=0, 
            full_std=True, 
            use_expln=False,
            squash_output=False, 
            features_extractor_class=features_extractor_class, 
            features_extractor_kwargs=None, 
            normalize_images=True, 
            optimizer_class=optimizer_class, 
            optimizer_kwargs=optimizer_kwargs
        )


class WeNeedToGoDeeper(ActorCriticPolicy):
    def __init__(
        self, 
        observation_space: gym.spaces.Space, 
        action_space: gym.spaces.Space, 
        lr_schedule: Schedule,
        use_sde=False,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=None
    ):

        net_arch = [128, 128, 64, dict(vf=[64, 32, 16], pi=[64, 32, 16])]
        features_extractor_class = NatureCNN
        activation_fn = th.nn.ReLU

        super().__init__(
            observation_space, 
            action_space, 
            lr_schedule, 
            net_arch, 
            activation_fn, 
            ortho_init=True, 
            use_sde=use_sde, 
            log_std_init=0, 
            full_std=True, 
            sde_net_arch=None,
            use_expln=False, 
            squash_output=False, 
            features_extractor_class=features_extractor_class, 
            features_extractor_kwargs=None, 
            normalize_images=True, 
            optimizer_class=optimizer_class, 
            optimizer_kwargs=optimizer_kwargs
        )
