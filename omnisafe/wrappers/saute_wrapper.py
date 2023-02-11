# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Environment wrapper for saute algorithms."""

from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces

from omnisafe.common.normalizer import Normalizer
from omnisafe.typing import NamedTuple, Optional
from omnisafe.utils.tools import as_tensor, expand_dims
from omnisafe.wrappers.cmdp_wrapper import CMDPWrapper
from omnisafe.wrappers.wrapper_registry import WRAPPER_REGISTRY


@dataclass
class RolloutLog:
    """Log for roll out."""

    ep_ret: np.ndarray
    ep_costs: np.ndarray
    ep_len: np.ndarray
    ep_budget: np.ndarray


@dataclass
class SauteData:
    """Data for Saute RL."""

    safety_budget: float
    unsafe_reward: float
    safety_obs: np.ndarray


@dataclass
class RolloutData:
    """Data for roll out."""

    local_steps_per_epoch: int
    max_ep_len: int
    use_cost: bool
    current_obs: torch.Tensor
    rollout_log: RolloutLog
    saute_data: SauteData


@WRAPPER_REGISTRY.register
class SauteWrapper(CMDPWrapper):
    r"""SauteEnvWrapper.

    Saute is a safe RL algorithm that uses state augmentation to ensure safety.
    The state augmentation is the concatenation of the original state and the safety state.
    The safety state is the safety budget minus the cost divided by the safety budget.

    .. note::
        - If the safety state is greater than 0, the reward is the original reward.
        - If the safety state is less than 0, the reward is the unsafe reward (always 0 or less than 0).

    ``omnisafe`` provides two implementations of Saute RL: :class:`PPOSaute` and :class:`PPOLagSaute`.

    References:

    - Title: Saute RL: Almost Surely Safe Reinforcement Learning Using State Augmentation
    - Authors: Aivar Sootla, Alexander I. Cowen-Rivers, Taher Jafferjee, Ziyan Wang,
      David Mguni, Jun Wang, Haitham Bou-Ammar.
    - URL: https://arxiv.org/abs/2202.06558
    """

    def __init__(self, env_id, cfgs: Optional[NamedTuple] = None, **env_kwargs) -> None:
        """Initialize environment wrapper.

        Args:
            env_id (str): environment id.
            cfgs (collections.namedtuple): configs.
            env_kwargs (dict): The additional parameters of environments.
        """
        super().__init__(env_id, cfgs, **env_kwargs)
        if hasattr(self.env, '_max_episode_steps'):
            max_ep_len = self.env._max_episode_steps
        else:
            max_ep_len = 1000
        if cfgs.scale_safety_budget:
            safety_budget = (
                cfgs.safety_budget
                * (1 - self.cfgs.saute_gamma**max_ep_len)
                / (1 - self.cfgs.saute_gamma)
                / np.float32(max_ep_len)
                * np.ones((self.cfgs.num_envs, 1))
            )
        else:
            safety_budget = cfgs.safety_budget * np.ones((self.cfgs.num_envs, 1))
        safety_obs = np.ones((self.cfgs.num_envs, 1), dtype=np.float32)
        self.rollout_data = RolloutData(
            0.0,
            max_ep_len,
            False,
            None,
            RolloutLog(
                np.zeros(self.cfgs.num_envs),
                np.zeros(self.cfgs.num_envs),
                np.zeros(self.cfgs.num_envs),
                np.zeros((self.cfgs.num_envs, 1)),
            ),
            SauteData(
                safety_budget=safety_budget,
                unsafe_reward=cfgs.unsafe_reward,
                safety_obs=safety_obs,
            ),
        )
        high = np.array(np.hstack([self.observation_space.high, np.inf]), dtype=np.float32)
        low = np.array(np.hstack([self.observation_space.low, np.inf]), dtype=np.float32)
        self.observation_space = spaces.Box(high=high, low=low)
        self.obs_normalizer = (
            Normalizer(shape=(self.cfgs.num_envs, self.observation_space.shape[0]), clip=5).to(
                self.cfgs.device
            )
            if self.cfgs.normalized_obs
            else None
        )
        self.rollout_data.current_obs = self.reset()[0]

    def augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Augmenting the obs with the safety obs.

        Detailedly, the augmented obs is the concatenation of the original obs and the safety obs.
        The safety obs is the safety budget minus the cost divided by the safety budget.

        Args:
            obs (np.ndarray): observation.
            safety_obs (np.ndarray): safety observation.
        """
        augmented_obs = np.hstack([obs, self.rollout_data.saute_data.safety_obs])
        return augmented_obs

    def safety_step(self, cost: np.ndarray, done: bool) -> np.ndarray:
        """Update the normalized safety obs.

        Args:
            cost (np.ndarray): cost.
        """
        if done:
            self.rollout_data.saute_data.safety_obs = np.ones(
                (self.cfgs.num_envs, 1), dtype=np.float32
            )
        else:
            self.rollout_data.saute_data.safety_obs -= (
                cost / self.rollout_data.saute_data.safety_budget
            )
            self.rollout_data.saute_data.safety_obs /= self.cfgs.saute_gamma

    def safety_reward(self, reward: np.ndarray) -> np.ndarray:
        """Update the reward.

        Args:
            reward (np.ndarray): reward.
            next_safety_obs (np.ndarray): next safety observation.
        """
        for idx, safety_obs in enumerate(self.rollout_data.saute_data.safety_obs):
            if safety_obs <= 0:
                reward[idx] = self.rollout_data.saute_data.unsafe_reward
        return reward

    def reset(self) -> tuple((torch.Tensor, dict)):
        """Reset environment.

        .. note::
            The safety obs is initialized to 1.0.

        Args:
            seed (int): seed for environment reset.
        """
        obs, info = self.env.reset()
        if self.cfgs.num_envs == 1:
            obs = expand_dims(obs)
            info = [info]
        self.rollout_data.saute_data.safety_obs = np.ones((self.cfgs.num_envs, 1), dtype=np.float32)
        obs = self.augment_obs(obs)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.cfgs.device), info

    def step(
        self, action: torch.Tensor
    ) -> tuple((torch.Tensor, torch.Tensor, torch.Tensor, bool, dict)):
        """Step environment.

        .. note::
            The safety obs is updated by the cost.
            The reward is updated by the safety obs.
            Detailedly, the reward is the original reward if the safety obs is greater than 0,
            otherwise the reward is the unsafe reward.

        Args:
            action (torch.Tensor): action.
        """
        next_obs, reward, cost, terminated, truncated, info = self.env.step(
            action.cpu().numpy().squeeze()
        )
        if self.cfgs.num_envs == 1:
            next_obs, reward, cost, terminated, truncated, info = expand_dims(
                next_obs, reward, cost, terminated, truncated, info
            )
            self.safety_step(cost, done=terminated | truncated)
            if terminated | truncated:
                augmented_obs, info = self.reset()
            else:
                augmented_obs = self.augment_obs(next_obs)
        else:
            augmented_obs = self.augment_obs(next_obs)
        self.rollout_data.rollout_log.ep_ret += reward
        self.rollout_data.rollout_log.ep_costs += cost
        self.rollout_data.rollout_log.ep_len += np.ones(self.cfgs.num_envs)
        self.rollout_data.rollout_log.ep_budget += self.rollout_data.saute_data.safety_obs
        reward = self.safety_reward(reward)
        return (
            as_tensor(augmented_obs, reward, cost, device=self.cfgs.device),
            terminated,
            truncated,
            info,
        )

    def reset_log(
        self,
        idx,
    ) -> None:
        (
            self.rollout_data.rollout_log.ep_ret[idx],
            self.rollout_data.rollout_log.ep_costs[idx],
            self.rollout_data.rollout_log.ep_len[idx],
            self.rollout_data.rollout_log.ep_budget[idx],
        ) = (0.0, 0.0, 0.0, 0.0)

    def rollout_log(
        self,
        logger,
        idx,
        is_train: bool = True,
    ) -> None:
        """Log the information of the rollout."""
        if is_train:
            logger.store(
                **{
                    'Metrics/EpRet': self.rollout_data.rollout_log.ep_ret[idx],
                    'Metrics/EpCost': self.rollout_data.rollout_log.ep_costs[idx],
                    'Metrics/EpLen': self.rollout_data.rollout_log.ep_len[idx],
                    'Metrics/EpBudget': self.rollout_data.rollout_log.ep_budget[idx],
                }
            )
        else:
            logger.store(
                **{
                    'Metrics/EpRet': self.rollout_data.rollout_log.ep_ret[idx],
                    'Metrics/EpCost': self.rollout_data.rollout_log.ep_costs[idx],
                    'Metrics/EpLen': self.rollout_data.rollout_log.ep_len[idx],
                    'Metrics/EpBudget': self.rollout_data.rollout_log.ep_budget[idx],
                }
            )
