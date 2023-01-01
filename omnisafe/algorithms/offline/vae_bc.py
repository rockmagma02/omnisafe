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
"""Implementation of the BCQ algorithm."""

import time
from copy import deepcopy

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.common.logger import Logger
from omnisafe.algorithms.offline.model import VAE
from omnisafe.utils.core import set_optimizer
from omnisafe.utils.offline_dataset import OfflineDataset
from omnisafe.wrappers import wrapper_registry


@registry.register
class VAEBC:  # pylint: disable=too-many-instance-attributes
    """Implementation of the VAEBC algorithm.

    References:
        Title: Off-Policy Deep Reinforcement Learning without Exploration
        Authors: Scott Fujimoto, David Meger, Doina Precup
        URL:  https://arxiv.org/abs/1812.02900
    """

    def __init__(self, env_id: str, cfgs=None) -> None:
        """Initialize the VAEBC algorithm.

        Args:
            env: The environment.
            cfgs: (default: :const:`None`)
                This is a dictionary of the algorithm hyper-parameters.
        """

        self.algo = self.__class__.__name__
        self.cfgs = deepcopy(cfgs)
        self.wrapper_type = self.cfgs.wrapper_type
        self.env = wrapper_registry.get(self.wrapper_type)(env_id)

        # set logger and save config
        self.logger = Logger(exp_name=cfgs.exp_name, data_dir=cfgs.data_dir, seed=cfgs.seed)
        self.logger.save_config(cfgs._asdict())
        # set seed
        seed = int(cfgs.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # setup model
        self.vae = VAE(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            hidden_sizes=cfgs.model_cfgs.vae_cfgs.hidden_sizes,
            latent_dim=cfgs.model_cfgs.vae_cfgs.latent_dim,
            activation=cfgs.model_cfgs.vae_cfgs.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode,
        )
        self.vae_optimizer = set_optimizer('Adam', module=self.vae, learning_rate=cfgs.vae_lr)
        # setup dataset
        self.dataset = OfflineDataset(cfgs.dataset_path, device=cfgs.device)
        self.obs_mean = self.dataset.obs_mean
        self.obs_std = self.dataset.obs_std

        self.change_device(cfgs.device)

        what_to_save = {
            'vae': self.vae,
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
        }
        self.logger.setup_torch_saver(what_to_save)
        self.logger.torch_save()

        self.start_time: float
        self.epoch_start_time: float
        self.epoch_step: int

    def learn(self):
        """Learn the VAE model."""
        self.logger.log('Start learning...')

        self.start_time = time.time()
        loader = self.dataset.get_loader(self.cfgs.batch_size)

        for epoch in range(self.cfgs.epochs):
            self.epoch_step = epoch
            self.epoch_start_time = time.time()

            # train
            for grad_step, batch in enumerate(loader):
                self._train(grad_step, batch)
                if grad_step >= self.cfgs.grad_steps_per_epoch:
                    break

            # evaluate
            self._evaluate()

            # log learning progress
            self._log()

            # save model
            if (epoch + 1) % self.cfgs.save_freq == 0:
                self.logger.torch_save()

        self.logger.log('Finish learning...')
        self.logger.close()

    def _train(self, grad_step, batch):
        """Train the VAE model.

        Args:
            grad_step (int): gradient step.
            batch (dict): batch data.
        """
        # train VAE
        self._update_vae(batch)

        if grad_step == self.cfgs.grad_steps_per_epoch - 1:
            self.logger.store(
                **{
                    'Epoch': self.epoch_step + 1,
                    'GradStep': (self.epoch_step + 1) * self.cfgs.grad_steps_per_epoch ,
                    'Time/Epoch': time.time() - self.epoch_start_time,
                    'Time/Total': time.time() - self.start_time,
                }
            )

    def _update_vae(self, batch):
        """Update the VAE model."""
        obs, act, reward, cost, done, next_obs = batch  # pylint: disable=unused-variable

        recon_loss, kl_loss = self.vae.loss(obs, act)
        loss = recon_loss + kl_loss
        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()

        self.logger.store(
            **{
                'Loss/Loss_vae': loss.item(),
                'Loss/Loss_recon': recon_loss.item(),
                'Loss/Loss_kl': kl_loss.item(),
            }
        )

    def _evaluate(self):
        # move model to cpu
        start_time = time.time()
        self.change_device('cpu')
        ep_reward, ep_cost = self.env.evaluate(self.agent_step, self.cfgs.eval_episodes)
        self.change_device(self.cfgs.device)

        self.logger.store(
            **{
                'Metrics/EpRet': ep_reward,
                'Metrics/EpCost': ep_cost,
                'Time/Eval': time.time() - start_time,
            }
        )

    def change_device(self, device):
        """Change the device of the model.

        Args:
            device (str): device name.
        """
        self.vae.to(device)
        self.obs_mean = self.obs_mean.to(device)
        self.obs_std = self.obs_std.to(device)

    def agent_step(self, obs):
        """Get action from the agent.

        Args:
            obs (np.ndarray): observation.

        Returns:
            np.ndarray: action.
        """
        obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        obs = obs.unsqueeze(0)
        action = self.vae.decode(obs).squeeze(0).detach().numpy()
        return action

    def _log(self):
        self.logger.log_tabular('Epoch')
        self.logger.log_tabular('GradStep')
        self.logger.log_tabular('Time/Total')
        self.logger.log_tabular('Time/Epoch')
        self.logger.log_tabular('Time/Eval')

        self.logger.log_tabular('Loss/Loss_vae')
        self.logger.log_tabular('Loss/Loss_recon')
        self.logger.log_tabular('Loss/Loss_kl')

        self.logger.log_tabular('Metrics/EpCost')
        self.logger.log_tabular('Metrics/EpRet')

        self.logger.dump_tabular()
