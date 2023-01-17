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
from omnisafe.algorithms.offline.bcq import BCQ
from omnisafe.algorithms.offline.model import VAE, BCQActor
from omnisafe.common.lagrange import Lagrange
from omnisafe.common.logger import Logger
from omnisafe.models import CriticBuilder
from omnisafe.utils.config_utils import namedtuple2dict
from omnisafe.utils.core import set_optimizer
from omnisafe.utils.offline_dataset import OfflineDataset
from omnisafe.wrappers import wrapper_registry


@registry.register
class BCQLag(BCQ, Lagrange):  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """Implementation of the BCQ Lag algorithm.

    References:
        Title: Constraints Penalized Q-learning for Safe Offline Reinforcement Learning
        Authors: Haoran Xu, Xianyuan Zhan, Xiangyu Zhu
        URL:  https://arxiv.org/abs/2107.09003
    """

    def __init__(self, env_id: str, cfgs=None) -> None:  # pylint: disable=super-init-not-called
        """Initialize the BCQ Lag algorithm.

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
        import os
        data_dir = os.path.join(cfgs.data_dir, cfgs.dataset_path[12:-4])
        self.logger = Logger(exp_name=cfgs.exp_name, data_dir=data_dir, seed=cfgs.seed)
        self.logger.save_config(namedtuple2dict(cfgs))
        # set seed
        seed = int(cfgs.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.set_seed(seed)
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
        self.criric = builder.build_critic('q', num_critics=2)
        self.target_criric = deepcopy(self.criric)
        self.critic_optimizer = set_optimizer(
            'Adam', module=self.criric, learning_rate=cfgs.critic_lr
        )
        self.cost_critic = builder.build_critic('q', num_critics=2)
        self.target_cost_critic = deepcopy(self.cost_critic)
        self.cost_critic_optimizer = set_optimizer(
            'Adam', module=self.cost_critic, learning_rate=cfgs.critic_lr
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
        self.cost_critic.to(cfgs.device)
        self.target_cost_critic.to(cfgs.device)
        self.actor.change_deivce(cfgs.device)
        self.target_actor.change_deivce(cfgs.device)

        Lagrange.__init__(
            self,
            cost_limit=cfgs.cost_limit,
            lagrangian_multiplier_init=cfgs.lagrangian_multiplier_init,
            lambda_lr=cfgs.lambda_lr,
            lambda_optimizer=cfgs.lambda_optimizer,
        )

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

        self._update_cost_critic(batch)

        # train actor
        self._update_actor(batch)

        self._soft_update_target()

        if grad_step == self.cfgs.grad_steps_per_epoch - 1:
            self.logger.store(
                **{
                    'Epoch': self.epoch_step + 1,
                    'GradStep': (self.epoch_step + 1) * self.cfgs.grad_steps_per_epoch,
                    'Time/Epoch': time.time() - self.epoch_start_time,
                    'Time/Total': time.time() - self.start_time,
                }
            )

    def _update_cost_critic(self, batch):  # pylint: disable=too-many-locals
        obs, act, reward, cost, done, next_obs = batch  # pylint: disable=unused-variable

        with torch.no_grad():
            # sample action from target actor
            next_obs_repeat = torch.repeat_interleave(next_obs, self.cfgs.sampled_action_num, dim=0)
            next_action = self.target_actor(next_obs_repeat, self.vae.decode(next_obs_repeat))

            # compute target cost
            qc1_target, qc2_target = self.target_cost_critic(next_obs_repeat, next_action)
            qc_target = self.cfgs.minimum_weighting * torch.min(qc1_target, qc2_target) + (
                1 - self.cfgs.minimum_weighting
            ) * torch.max(qc1_target, qc2_target)
            qc_target = qc_target.reshape(self.cfgs.batch_size, -1).max(dim=1)[0].reshape(-1, 1)
            qc_target = (
                self.cfgs.cost_scale * cost.unsqueeze(1)
                + (1 - done.unsqueeze(1)) * self.cfgs.gamma * qc_target
            )
            qc_target = qc_target.squeeze(1)

        # train cost critic
        qc1, qc2 = self.cost_critic(obs, act)
        cost_critic_loss = nn.functional.mse_loss(qc1, qc_target) + nn.functional.mse_loss(
            qc2, qc_target
        )
        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()

        self.logger.store(
            **{
                'Loss/Loss_cost_critic': cost_critic_loss.item(),
                'Qc/data_qc': qc1[0].mean().item(),
                'Qc/target_qc': qc_target[0].mean().item(),
            }
        )

    def _update_actor(self, batch):
        obs, act, reward, cost, done, next_obs = batch  # pylint: disable=unused-variable

        # actor training
        action = self.actor(obs, self.vae.decode(obs))
        qr_curr = self.criric(obs, action)[0]
        qc_curr = self.cost_critic(obs, action)[0]
        actor_loss = -(qr_curr.mean() - self.lagrangian_multiplier * qc_curr.mean())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.epoch_step > self.cfgs.begin_update_lag_epoch:
            self.update_lagrange_multiplier(qc_curr.mean().item() / self.cfgs.cost_scale)

        self.logger.store(
            **{
                'Qr/current_Qr': qr_curr[0].mean().item(),
                'Qc/current_Qc': qc_curr[0].mean().item(),
                'Loss/Loss_actor': actor_loss.item(),
                'Misc/LagrangeMultiplier': self.lagrangian_multiplier.item(),
            }
        )

    def _soft_update_target(self) -> None:
        super()._soft_update_target()
        for target_param, param in zip(
            self.target_cost_critic.parameters(), self.cost_critic.parameters()
        ):
            target_param.data.copy_(
                self.cfgs.smooth_coeff * param.data
                + (1 - self.cfgs.smooth_coeff) * target_param.data
            )

    def _log(self):
        self.logger.log_tabular('Loss/Loss_cost_critic')
        self.logger.log_tabular('Qc/data_qc')
        self.logger.log_tabular('Qc/target_qc')
        self.logger.log_tabular('Qc/current_Qc')
        self.logger.log_tabular('Misc/LagrangeMultiplier')
        super()._log()
