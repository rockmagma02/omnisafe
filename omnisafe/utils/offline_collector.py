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
"""A collector help to collect offline data"""

import json
import os
from typing import Literal

import numpy as np
import torch
import tqdm
from gymnasium.spaces import Box, Discrete

from omnisafe.models import ActorBuilder
from omnisafe.utils.core import combined_shape
from omnisafe.utils.online_mean_std import OnlineMeanStd
from omnisafe.wrappers import OfflineEnvWrapper as Env


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals


class Collector:
    """A collector help to collect offline data"""

    def __init__(
        self,
        env_id: str,
        size: int = 1_000_000,
        random_proportion: float = 0.0,
        expert_proportion: float = 0.5,
        expert_path: str = '',
        expert_model_name: str = 'best_model.pt',
        unsafe_proportion: float = 0.5,
        unsafe_path: str = '',
        unsafe_model_name: str = 'best_model.pt',
        noise_std: float = 0.3,
        cost_limit: float = 30.0,
    ) -> None:
        """A collector help to collect offline data

        Args:
            env_id (str): environment id
            size (int, optional): number of samples. Defaults to 1_000_000.
            random_proportion (float, optional): proportion of random data. Defaults to 0.0.
            expert_proportion (float, optional): proportion of expert data. Defaults to 0.5.
            expert_path (str, optional): expert model path. Defaults to ''.
            expert_model_name (str, optional): expert model name. Defaults to 'best_model.pt'.
            unsafe_proportion (float, optional): proportion of unsafe data. Defaults to 0.5.
            unsafe_path (str, optional): unsafe model path. Defaults to ''.
            unsafe_model_name (str, optional): unsafe model name. Defaults to 'best_model.pt'.
            noise_std (float, optional): noise std. Defaults to 0.3.
        """
        self.env_id = env_id
        self.size = size

        self.random_proportion = random_proportion

        self.expert_proportion = expert_proportion
        self.expert_path = expert_path
        self.expert_model_name = expert_model_name

        self.unsafe_proportion = unsafe_proportion
        self.unsafe_path = unsafe_path
        self.unsafe_model_name = unsafe_model_name

        self.noise_std = noise_std
        self.cost_limit = cost_limit

        assert self.random_proportion + self.expert_proportion + self.unsafe_proportion == 1.0

        env = Env(self.env_id)
        self.max_ep_len = env.max_ep_len
        self._obs_dim = env.observation_space.shape
        self._act_dim = env.action_space.shape

        self._obs = np.zeros(combined_shape(size, self._obs_dim), dtype=np.float32)
        self._action = np.zeros(combined_shape(size, self._act_dim), dtype=np.float32)
        self._reward = np.zeros(size, dtype=np.float32)
        self._cost = np.zeros(size, dtype=np.float32)
        self._done = np.zeros(size, dtype=np.float32)
        self._next_obs = np.zeros(combined_shape(size, self._obs_dim), dtype=np.float32)

        ptr: int = 0
        self.random_slice = None
        self.expert_slice = None
        self.unsafe_slice = None
        if self.random_proportion > 0:
            self.random_size = int(self.size * self.random_proportion)
            self.random_slice = slice(ptr, ptr + self.random_size)
            ptr += self.random_size
        if self.expert_proportion > 0:
            self.expert_size = int(self.size * self.expert_proportion)
            self.expert_slice = slice(ptr, ptr + self.expert_size)
            ptr += self.expert_size
        if self.unsafe_proportion > 0:
            self.unsafe_slice = slice(ptr, self.size)
            self.unsafe_size = self.size - ptr

    def collect(self):
        """Collect data"""
        progress_bar = tqdm.tqdm(total=self.size)
        ave_total_reward, ave_total_cost = 0, 0
        if self.random_proportion > 0:
            ave_reward, ave_cost = 0, 0
            env, agent_step = self._load_env_agent('random')
            for ptr in range(self.random_slice.start, self.random_slice.stop, self.max_ep_len):
                data = self._roll_out_eposide(env, agent_step)
                self._obs[ptr:ptr + self.max_ep_len] = data['obs']
                self._action[ptr:ptr + self.max_ep_len] = data['action']
                self._reward[ptr:ptr + self.max_ep_len] = data['reward']
                self._cost[ptr:ptr + self.max_ep_len] = data['cost']
                self._done[ptr:ptr + self.max_ep_len] = data['done']
                self._next_obs[ptr:ptr + self.max_ep_len] = data['next_obs']
                progress_bar.update(self.max_ep_len)
                ave_reward += data['ep_ret']
                ave_cost += data['ep_cost']
            ave_reward /= (self.random_size / self.max_ep_len)
            ave_cost /= (self.random_size / self.max_ep_len)
            ave_total_reward += (ave_reward * self.random_proportion)
            ave_total_cost += (ave_cost * self.random_proportion)
            print(f'random data: ave_reward: {ave_reward:.2f}, ave_cost: {ave_cost:.2f}')

        if self.expert_proportion > 0:
            env, agent_step = self._load_env_agent('expert')
            ave_reward, ave_cost = 0, 0
            for ptr in range(self.expert_slice.start, self.expert_slice.stop, self.max_ep_len):
                data = self._roll_out_eposide(env, agent_step)
                while data['ep_cost'] > self.cost_limit:
                    data = self._roll_out_eposide(env, agent_step)
                self._obs[ptr:ptr + self.max_ep_len] = data['obs']
                self._action[ptr:ptr + self.max_ep_len] = data['action']
                self._reward[ptr:ptr + self.max_ep_len] = data['reward']
                self._cost[ptr:ptr + self.max_ep_len] = data['cost']
                self._done[ptr:ptr + self.max_ep_len] = data['done']
                self._next_obs[ptr:ptr + self.max_ep_len] = data['next_obs']
                progress_bar.update(self.max_ep_len)
                ave_reward += data['ep_ret']
                ave_cost += data['ep_cost']
            ave_reward /= (self.expert_size / self.max_ep_len)
            ave_cost /= (self.expert_size / self.max_ep_len)
            ave_total_reward += (ave_reward * self.expert_proportion)
            ave_total_cost += (ave_cost * self.expert_proportion)
            print(f'expert data: ave_reward: {ave_reward:.2f}, ave_cost: {ave_cost:.2f}')

        if self.unsafe_proportion > 0:
            safe_env, safe_agent_step = self._load_env_agent('expert')
            unsafe_env, unsafe_agent_step = self._load_env_agent('unsafe')
            use_safe = True
            ave_reward, ave_cost = 0, 0
            for ptr in range(self.unsafe_slice.start, self.unsafe_slice.stop, self.max_ep_len):
                if use_safe:
                    data = self._roll_out_eposide(safe_env, safe_agent_step)
                    while data['ep_cost'] <= self.cost_limit:
                        data = self._roll_out_eposide(safe_env, safe_agent_step)
                    use_safe = False
                else:
                    data = self._roll_out_eposide(unsafe_env, unsafe_agent_step)
                    use_safe = True
                self._obs[ptr:ptr + self.max_ep_len] = data['obs']
                self._action[ptr:ptr + self.max_ep_len] = data['action']
                self._reward[ptr:ptr + self.max_ep_len] = data['reward']
                self._cost[ptr:ptr + self.max_ep_len] = data['cost']
                self._done[ptr:ptr + self.max_ep_len] = data['done']
                self._next_obs[ptr:ptr + self.max_ep_len] = data['next_obs']
                progress_bar.update(self.max_ep_len)
                ave_reward += data['ep_ret']
                ave_cost += data['ep_cost']
            ave_reward /= (self.unsafe_size / self.max_ep_len)
            ave_cost /= (self.unsafe_size / self.max_ep_len)
            ave_total_reward += (ave_reward * self.unsafe_proportion)
            ave_total_cost += (ave_cost * self.unsafe_proportion)
            print(f'unsafe data: ave_reward: {ave_reward:.2f}, ave_cost: {ave_cost:.2f}')

        print(f'ave_total_reward: {ave_total_reward:.2f}, ave_total_cost: {ave_total_cost:.2f}')


    def save(self, save_dir: str, save_name: str):
        """save data

        Args:
            save_dir (str): save directory
            save_name (str): save name
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_name is None:
            save_name = 'data.npz'
        else:
            save_name = f'{save_name}.npz'
        np.savez(
            os.path.join(save_dir, save_name),
            obs=self._obs,
            action=self._action,
            reward=self._reward,
            cost=self._cost,
            done=self._done,
            next_obs=self._next_obs,
        )

    def _roll_out_eposide(self, env, agent_step: callable):
        obs_lst = []
        act_lst = []
        reward_lst = []
        cost_lst = []
        done_lst = []
        next_obs_lst = []

        done = False
        obs, _ = env.reset()
        ep_ret, ep_costs, ep_len = 0.0, 0.0, 0
        while not done:
            act = agent_step(torch.as_tensor(obs, dtype=torch.float32))
            next_obs, reward, cost, terminated, truncated, _ = env.step(act)

            if terminated or truncated: # for fixed length rollout
                done = True

            obs_lst.append(obs)
            act_lst.append(act)
            reward_lst.append(reward)
            cost_lst.append(cost)
            done_lst.append(float(done))
            next_obs_lst.append(next_obs)

            ep_ret += reward
            ep_costs += cost
            ep_len += 1

            obs = next_obs

        return {
            'obs': np.array(obs_lst),
            'action': np.array(act_lst),
            'reward': np.array(reward_lst),
            'cost': np.array(cost_lst),
            'done': np.array(done_lst),
            'next_obs': np.array(next_obs_lst),
            'ep_ret': ep_ret,
            'ep_cost': ep_costs,
            'ep_len': ep_len,
        }

    def _load_env_agent(self, agent_type: Literal['random', 'expert', 'unsafe']):
        # load env
        env = Env(self.env_id)

        action_space = env.action_space
        observation_space = env.env.observation_space
        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            act_dim = action_space.shape[0]
            actor_type = 'gaussian_annealing'
            act_min = torch.as_tensor(action_space.low, dtype=torch.float32)
            act_max = torch.as_tensor(action_space.high, dtype=torch.float32)
        elif isinstance(action_space, Discrete):
            act_dim = action_space.n
            actor_type = 'categorical'
            act_min, act_max = None, None

        if agent_type == 'random':
            actor_builder = ActorBuilder(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_sizes=[64, 64],
                activation='tanh',
                weight_initialization_mode='kaiming_uniform',
            )
            actor = actor_builder.build_actor(
                actor_type, act_min=act_min, act_max=act_max, start_std=self.noise_std
            )

            def agent_step(obs):
                return actor.predict(obs).numpy()

        else:
            # load path
            path = self.expert_path if agent_type == 'expert' else self.unsafe_path

            # load config
            cfg_path = os.path.join(path, 'config.json')
            try:
                with open(cfg_path, 'r', encoding='utf-8') as file:
                    cfg = json.load(file)
            except FileNotFoundError as exception:
                raise FileNotFoundError(
                    'The config file is not found in the save directory.'
                ) from exception

            # load the saved model
            model_name = (
                self.expert_model_name if agent_type == 'expert' else self.unsafe_model_name
            )
            model_path = os.path.join(path, 'torch_save', model_name)
            try:
                model_params = torch.load(model_path)
            except FileNotFoundError as exception:
                raise FileNotFoundError(
                    'The model is not found in the save directory.'
                ) from exception

            actor_config = cfg['model_cfgs']['ac_kwargs']['pi']
            weight_initialization_mode = cfg['model_cfgs']['weight_initialization_mode']
            actor_builder = ActorBuilder(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_sizes=actor_config['hidden_sizes'],
                activation=actor_config['activation'],
                weight_initialization_mode=weight_initialization_mode,
            )
            actor = actor_builder.build_actor(
                actor_type=actor_config['actor_type'],
                act_min=act_min,
                act_max=act_max,
                start_std=self.noise_std,
            )
            actor.load_state_dict(model_params['pi'])

            # make the observation OMS
            if cfg['standardized_obs']:
                obs_oms = OnlineMeanStd(shape=observation_space.shape)
                obs_oms.load_state_dict(model_params['obs_oms'])
            else:

                def obs_oms(obs):
                    return obs

            def model_step(obs):
                with torch.no_grad():
                    obs = obs_oms(obs)
                    action = actor.predict(obs, deterministic=False)
                return action.numpy()

            agent_step = model_step

        return env, agent_step
