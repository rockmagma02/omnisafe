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
from torch import nn

from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.vae_bc import VAEBC
from omnisafe.algorithms.offline.model import VAE, BCQActor
from omnisafe.common.logger import Logger
from omnisafe.models import CriticBuilder
from omnisafe.utils.core import set_optimizer
from omnisafe.utils.offline_dataset import OfflineDataset
from omnisafe.wrappers import wrapper_registry


@registry.register
class BCQ(VAEBC):  # pylint: disable=too-many-instance-attributes, too-few-public-methods
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
        self.actor = BCQActor(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            act_min=torch.as_tensor(self.env.action_space.low, dtype=torch.float32),
            act_max=torch.as_tensor(self.env.action_space.high, dtype=torch.float32),
            hidden_sizes=cfgs.model_cfgs.actor_cfgs.hidden_sizes,
            activation=cfgs.model_cfgs.actor_cfgs.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode,
            phi=cfgs.phi,
        )
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = set_optimizer('Adam', module=self.actor, learning_rate=cfgs.actor_lr)
        builder = CriticBuilder(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            hidden_sizes=cfgs.model_cfgs.critic_cfgs.hidden_sizes,
            activation=cfgs.model_cfgs.critic_cfgs.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode,
            shared=None,
        )
        self.criric = builder.build_critic('q', num_critics = 2)
        self.target_criric = deepcopy(self.criric)
        self.critic_optimizer = set_optimizer(
            'Adam', module=self.criric, learning_rate=cfgs.critic_lr
        )

        # setup dataset
        self.dataset = OfflineDataset(cfgs.dataset_path, device=cfgs.device)
        self.obs_mean = self.dataset.obs_mean
        self.obs_std = self.dataset.obs_std

        self.change_device(cfgs.device)

        # move model to device
        self.vae.to(cfgs.device)
        self.criric.to(cfgs.device)
        self.target_criric.to(cfgs.device)
        self.actor.change_deivce(cfgs.device)
        self.target_actor.change_deivce(cfgs.device)

        what_to_save = {
            'vae': self.vae,
            'actor': self.actor,
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
        }
        self.logger.setup_torch_saver(what_to_save)
        self.logger.torch_save()

    def _train(self, grad_step, batch):
        """Train the VAE model.

        Args:
            grad_step (int): gradient step.
            batch (dict): batch data.
        """
        # train VAE
        self._update_vae(batch)

        # train critic
        self._update_critic(batch)

        # train actor
        self._update_actor(batch)

        self._soft_update_target()

        if grad_step == self.cfgs.grad_steps_per_epoch - 1:
            self.logger.store(
                **{
                    'Epoch': self.epoch_step + 1,
                    'GradStep': (self.epoch_step + 1) * self.cfgs.grad_steps_per_epoch ,
                    'Time/Epoch': time.time() - self.epoch_start_time,
                    'Time/Total': time.time() - self.start_time,
                }
            )

    def _update_critic(self, batch):  # pylint: disable=too-many-locals
        obs, act, reward, cost, done, next_obs = batch  # pylint: disable=unused-variable

        with torch.no_grad():
            # sample action from actor
            next_obs_repeat = torch.repeat_interleave(next_obs, self.cfgs.sampled_action_num, dim=0)
            next_action = self.target_actor(next_obs_repeat, self.vae.decode(next_obs_repeat))

            # compute target q
            qr1_target, qr2_target = self.target_criric(next_obs_repeat, next_action)
            qr_target = self.cfgs.minimum_weighting * torch.min(qr1_target, qr2_target) + (
                1 - self.cfgs.minimum_weighting
            ) * torch.max(qr1_target, qr2_target)
            qr_target = qr_target.reshape(self.cfgs.batch_size, -1).max(dim=1)[0].reshape(-1, 1)
            qr_target = (
                self.cfgs.reward_scale * reward.unsqueeze(1)
                + (1 - done.unsqueeze(1)) * self.cfgs.gamma * qr_target
            )
            qr_target = qr_target.squeeze(1)

        qr1, qr2 = self.criric(obs, act)
        critic_loss = nn.functional.mse_loss(qr1, qr_target) + nn.functional.mse_loss(qr2, qr_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.logger.store(
            **{
                'Loss/Loss_critic': critic_loss.item(),
                'Qr/data_Qr': qr1[0].mean().item(),
                'Qr/target_Qr': qr_target[0].mean().item(),
            }
        )

    def _update_actor(self, batch):
        obs, act, reward, cost, done, next_obs = batch  # pylint: disable=unused-variable

        # actor training
        action = self.actor(obs, self.vae.decode(obs))
        qr_curr = self.criric(obs, action)[0]
        actor_loss = -qr_curr.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.logger.store(
            **{
                'Qr/current_Qr': qr_curr[0].mean().item(),
                'Loss/Loss_actor': actor_loss.item(),
            }
        )

    def _soft_update_target(self) -> None:
        for target_param, param in zip(self.target_criric.parameters(), self.criric.parameters()):
            target_param.data.copy_(
                self.cfgs.smooth_coeff * param.data
                + (1 - self.cfgs.smooth_coeff) * target_param.data
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                self.cfgs.smooth_coeff * param.data
                + (1 - self.cfgs.smooth_coeff) * target_param.data
            )

    def change_device(self, device):
        super().change_device(device)
        self.actor.change_deivce(device)

    def agent_step(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        obs = obs.unsqueeze(0)
        act = self.vae.decode(obs)
        act = self.actor(obs, act).squeeze(0).detach().numpy()
        return act

    def _log(self):
        self.logger.log_tabular('Loss/Loss_actor')
        self.logger.log_tabular('Loss/Loss_critic')

        self.logger.log_tabular('Qr/current_Qr')
        self.logger.log_tabular('Qr/data_Qr')
        self.logger.log_tabular('Qr/target_Qr')
        super()._log()
