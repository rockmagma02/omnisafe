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
"""A dataset class for offline training."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# pylint: disable=too-many-instance-attributes


class OfflineDataset(Dataset):
    """A dataset class for offline training."""

    def __init__(
        self,
        data_path: str,
        device: str,
        obs_standardization: bool = True,
    ):
        """Initialize the dataset.

        Args:
            data_path: The path to the data.
            device: The device to load the data to.
        """
        print('Loading data from', data_path)

        try:
            data = np.load(data_path)
        except FileNotFoundError as exception:
            raise FileNotFoundError(f'File {data_path} not found.') from exception

        obs = data['obs']
        action = data['action']
        reward = data['reward']
        cost = data['cost']
        done = data['done']
        next_obs = data['next_obs']

        self.obs = torch.from_numpy(obs).float().to(device)
        self.action = torch.from_numpy(action).float().to(device)
        self.reward = torch.from_numpy(reward).float().to(device)
        self.cost = torch.from_numpy(cost).float().to(device)
        self.done = torch.from_numpy(done).float().to(device)
        self.next_obs = torch.from_numpy(next_obs).float().to(device)

        self.length = self.obs.shape[0]

        print('Data loaded.')

        print('Standardizing data.')
        if obs_standardization:
            self.obs_mean = self.obs.mean(dim=0)
            self.obs_std = self.obs.std(dim=0)
            self.obs = (self.obs - self.obs_mean) / (self.obs_std + 1e-8)
            self.next_obs = (self.next_obs - self.obs_mean) / (self.obs_std + 1e-8)
        else:
            self.obs_mean = torch.zeros_like(self.obs[0])
            self.obs_std = torch.ones_like(self.obs[0])  # for the case of no standardization
        print(
            f'Data standardized, the mean of obs is {self.obs_mean},\n the std of obs is {self.obs_std},'
        )

    def __len__(self):
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Return the data and label at the given index."""
        obs = self.obs[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        cost = self.cost[idx]
        done = self.done[idx]
        next_obs = self.next_obs[idx]

        return obs, action, reward, cost, done, next_obs

    def get_loader(self, batch_size: int, shuffle: bool = True, infinite: bool = True):
        """Return a data loader for the dataset.

        Args:
            batch_size: The batch size for the data loader.
            shuffle: Whether to shuffle the dataset.

        Returns:
            A data loader for the dataset.
        """
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=True)

        def infinite_dataloader():
            while True:
                for data in dataloader:
                    yield data

        if infinite:
            return infinite_dataloader()
        else:
            return dataloader


class OfflineDatasetWithInitObs(OfflineDataset):
    """A dataset class for offline training with initial observation."""

    def __init__(  # pylint: disable=super-init-not-called
        self,
        data_path: str,
        device: str,
        obs_standardization: bool = True,
        episode_length: int = 500,
    ):
        """Initialize the dataset.

        Args:
            data_path: The path to the data.
            device: The device to load the data to.
        """
        print('Loading data from', data_path)

        try:
            data = np.load(data_path)
        except FileNotFoundError as exception:
            raise FileNotFoundError(f'File {data_path} not found.') from exception

        obs = data['obs']
        action = data['action']
        reward = data['reward']
        cost = data['cost']
        done = data['done']
        next_obs = data['next_obs']

        if 'init_obs' in data.keys():
            init_obs = data['init_obs']
        else:
            init_obs = obs[::episode_length]
            init_obs = np.repeat(init_obs, episode_length, axis=0)

        self.obs = torch.from_numpy(obs).float().to(device)
        self.action = torch.from_numpy(action).float().to(device)
        self.reward = torch.from_numpy(reward).float().to(device)
        self.cost = torch.from_numpy(cost).float().to(device)
        self.done = torch.from_numpy(done).float().to(device)
        self.next_obs = torch.from_numpy(next_obs).float().to(device)
        self.init_obs = torch.from_numpy(init_obs).float().to(device)

        self.length = self.obs.shape[0]

        print('Data loaded.')

        print('Standardizing data.')
        if obs_standardization:
            self.obs_mean = self.obs.mean(dim=0)
            self.obs_std = self.obs.std(dim=0)
            self.obs = (self.obs - self.obs_mean) / (self.obs_std + 1e-8)
            self.next_obs = (self.next_obs - self.obs_mean) / (self.obs_std + 1e-8)
            self.init_obs = (self.init_obs - self.obs_mean) / (self.obs_std + 1e-8)
        else:
            self.obs_mean = torch.zeros_like(self.obs[0])
            self.obs_std = torch.ones_like(self.obs[0])
        print(
            f'Data standardized, the mean of obs is {self.obs_mean},\n the std of obs is {self.obs_std},'
        )

    def __getitem__(self, idx):
        """Return the data and label at the given index."""
        obs = self.obs[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        cost = self.cost[idx]
        done = self.done[idx]
        next_obs = self.next_obs[idx]
        init_obs = self.init_obs[idx]

        return obs, action, reward, cost, done, next_obs, init_obs


class OfflineDatasetWithSafeFlag(OfflineDataset):
    """A dataset class for offline training with initial observation."""

    def __init__(  # pylint: disable=super-init-not-called
        self,
        data_path: str,
        device: str,
        obs_standardization: bool = True,
        episode_length: int = 500,
    ):
        """Initialize the dataset.

        Args:
            data_path: The path to the data.
            device: The device to load the data to.
        """
        print('Loading data from', data_path)

        try:
            data = np.load(data_path)
        except FileNotFoundError as exception:
            raise FileNotFoundError(f'File {data_path} not found.') from exception

        obs = data['obs']
        action = data['action']
        reward = data['reward']
        cost = data['cost']
        done = data['done']
        next_obs = data['next_obs']

        if 'is_safe' in data.keys():
            is_safe = data['is_safe']
        else:
            is_safe1 = np.full((1_000_000,), 1.)
            is_safe2 = np.full((1_000_000,), 0.)
            is_safe = np.concatenate((is_safe1, is_safe2), axis=0)

        self.obs = torch.from_numpy(obs).float().to(device)
        self.action = torch.from_numpy(action).float().to(device)
        self.reward = torch.from_numpy(reward).float().to(device)
        self.cost = torch.from_numpy(cost).float().to(device)
        self.done = torch.from_numpy(done).float().to(device)
        self.next_obs = torch.from_numpy(next_obs).float().to(device)
        self.is_safe = torch.from_numpy(is_safe).float().to(device)

        self.length = self.obs.shape[0]

        print('Data loaded.')

        print('Standardizing data.')
        if obs_standardization:
            self.obs_mean = self.obs.mean(dim=0)
            self.obs_std = self.obs.std(dim=0)
            self.obs = (self.obs - self.obs_mean) / (self.obs_std + 1e-8)
            self.next_obs = (self.next_obs - self.obs_mean) / (self.obs_std + 1e-8)
        else:
            self.obs_mean = torch.zeros_like(self.obs[0])
            self.obs_std = torch.ones_like(self.obs[0])
        print(
            f'Data standardized, the mean of obs is {self.obs_mean},\n the std of obs is {self.obs_std},'
        )

    def __getitem__(self, idx):
        """Return the data and label at the given index."""
        obs = self.obs[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        cost = self.cost[idx]
        done = self.done[idx]
        next_obs = self.next_obs[idx]
        is_safe = self.is_safe[idx]

        return obs, action, reward, cost, done, next_obs, is_safe
