# Copyright 2022 OmniSafe Team. All Rights Reserved.
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
"""env wrapper for offline algorithms"""

from typing import Callable

import numpy as np
import safety_gymnasium

from omnisafe.wrappers.wrapper_registry import WRAPPER_REGISTRY


@WRAPPER_REGISTRY.register
class OfflineEnvWrapper:  # pylint: disable=too-many-instance-attributes
    """env_wrapper for offline algorithms"""

    def __init__(self, env_id):
        # check env_id is str
        self.env = safety_gymnasium.make(env_id)
        self.env_id = env_id
        self.metadata = self.env.metadata

        if hasattr(self.env, '_max_episode_steps'):
            self.max_ep_len = self.env._max_episode_steps
        else:
            self.max_ep_len = 500

        if env_id in ['SafetyPointGoal1-v0', 'SafetyCarGoal1-v0']:
            self.max_ep_len = 1000
        else:
            self.max_ep_len = 500

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.seed = None
        self.curr_o = self.env.reset(seed=self.seed)

    def make(self):
        """create environments"""
        return self.env

    def reset(self, seed=None):
        """reset environment"""
        self.curr_o, info = self.env.reset(seed=seed)
        return self.curr_o, info

    def set_seed(self, seed):
        """set environment seed"""
        self.seed = seed

    def step(self, action):
        """engine step"""
        next_obs, reward, cost, terminated, truncated, info = self.env.step(action)
        return next_obs, reward, cost, terminated, truncated, info

    def evaluate(self, agent_step: Callable[[np.ndarray], np.ndarray], num_episodes: int):
        """evaluate agent"""
        total_reward = 0
        total_cost = 0
        for _ in range(num_episodes):
            self.reset()
            done = False
            while not done:
                action = agent_step(self.curr_o)
                self.curr_o, reward, cost, terminated, truncated, _ = self.step(action)
                total_reward += reward
                total_cost += cost
                done = terminated or truncated
        return total_reward / num_episodes, total_cost / num_episodes
