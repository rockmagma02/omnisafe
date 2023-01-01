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
from omnisafe.algorithms.offline.crr import CRR
from omnisafe.common.lagrange import Lagrange
from omnisafe.common.logger import Logger
from omnisafe.models import ActorBuilder, CriticBuilder
from omnisafe.utils.core import set_optimizer
from omnisafe.utils.offline_dataset import OfflineDataset
from omnisafe.wrappers import wrapper_registry


@registry.register
class CCRR(CRR, Lagrange):  # pylint: disable=too-many-instance-attributes
    """Implementation of the C-CRR algorithm.

    References:
        Title: COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation
        Authors: Jongmin Lee, Cosmin Paduraru, Daniel J. Mankowitz, Nicolas Heess,
                    Doina Precup, Kee-Eung Kim, Arthur Guez
        URL:  https://arxiv.org/abs/2204.08957
    """

    def __init__(self, env_id: str, cfgs=None) -> None:  # pylint: disable=super-init-not-called
        """Initialize the C-CRR algorithm.

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
        self.env.set_seed(seed)
        # setup model
        builder = ActorBuilder(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            hidden_sizes=cfgs.model_cfgs.actor_cfgs.hidden_sizes,
            activation=cfgs.model_cfgs.actor_cfgs.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode,
        )
        self.actor = builder.build_actor(
            actor_type='gaussian_learning',
            act_min=torch.as_tensor(self.env.action_space.low, dtype=torch.float32),
            act_max=torch.as_tensor(self.env.action_space.high, dtype=torch.float32),
        )
        self.actor_optimizer = set_optimizer('Adam', module=self.actor, learning_rate=cfgs.actor_lr)
        builder = CriticBuilder(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            hidden_sizes=cfgs.model_cfgs.critic_cfgs.hidden_sizes,
            activation=cfgs.model_cfgs.critic_cfgs.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode,
        )
        self.critic = builder.build_critic('q', num_critics=2)
        self.target_critic = deepcopy(self.critic)
        self.critic_optimizer = set_optimizer(
            'Adam', module=self.critic, learning_rate=cfgs.critic_lr
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

        self.actor.change_device(cfgs.device)
        self.critic.to(cfgs.device)
        self.target_critic.to(cfgs.device)
        self.cost_critic.to(cfgs.device)
        self.target_cost_critic.to(cfgs.device)

        Lagrange.__init__(
            self,
            cost_limit=cfgs.cost_limit,
            lagrangian_multiplier_init=cfgs.lagrangian_multiplier_init,
            lambda_lr=cfgs.lambda_lr,
            lambda_optimizer=cfgs.lambda_optimizer,
        )

        what_to_save = {
            'actor': self.actor,
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
        }
        self.logger.setup_torch_saver(what_to_save)
        self.logger.torch_save()

    def _train(self, grad_step, batch):
        # train critic
        self._update_critic(batch)

        self._updata_cost_critic(batch)

        self._soft_update_target()

        # train actor
        self._update_actor(batch)

        if grad_step == self.cfgs.grad_steps_per_epoch - 1:
            self.logger.store(
                **{
                    'Epoch': self.epoch_step + 1,
                    'GradStep': (self.epoch_step + 1) * self.cfgs.grad_steps_per_epoch,
                    'Time/Epoch': time.time() - self.epoch_start_time,
                    'Time/Total': time.time() - self.start_time,
                }
            )

    def _updata_cost_critic(self, batch):  # pylint: disable=too-many-locals
        obs, act, reward, cost, done, next_obs = batch  # pylint: disable=unused-variable

        # compute target
        with torch.no_grad():
            next_action = self.actor.predict(next_obs, deterministic=False)
            qc1_target, qc2_target = self.target_cost_critic(next_obs, next_action)
            qc_target = torch.min(qc1_target, qc2_target)
            qc_target = self.cfgs.cost_scale * cost + self.cfgs.gamma * (1 - done) * qc_target

        # compute loss and back-prop
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
                'Qc/data_Qc': qc1[0].mean().item(),
                'Qc/target_Qc': qc_target[0].mean().item(),
            }
        )

    def _update_actor(self, batch):  # pylint: disable=too-many-locals
        obs, act, reward, cost, done, next_obs = batch  # pylint: disable=unused-variable

        qr1, qr2 = self.critic(obs, act)
        qr_data = torch.min(qr1, qr2)

        qc1, qc2 = self.cost_critic(obs, act)
        qc_data = torch.min(qc1, qc2)

        obs_repeat = (
            obs.unsqueeze(1)
            .repeat(1, self.cfgs.sampled_action_num, 1)
            .view(obs.shape[0] * self.cfgs.sampled_action_num, obs.shape[1])
        )
        act_sample = self.actor.predict(obs_repeat, deterministic=False)

        qr1_sample, qr2_sample = self.critic(obs_repeat, act_sample)
        qr_sample = torch.min(qr1_sample, qr2_sample)
        mean_qr = torch.vstack(
            [q.mean() for q in qr_sample.reshape(-1, self.cfgs.sampled_action_num, 1)]
        )
        adv_r = qr_data - mean_qr.squeeze(1)

        qc1_sample, qc2_sample = self.cost_critic(obs_repeat, act_sample)
        qc_sample = torch.min(qc1_sample, qc2_sample)
        mean_qc = torch.vstack(
            [q.mean() for q in qc_sample.reshape(-1, self.cfgs.sampled_action_num, 1)]
        )
        adv_c = qc_data - mean_qc.squeeze(1)

        exp_adv = torch.exp((adv_r - self.lagrangian_multiplier * adv_c).detach() / self.cfgs.beta)

        dist, log_p = self.actor(obs, act)  # pylint: disable=unused-variable
        bc_loss = -log_p
        policy_loss = (exp_adv * bc_loss).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.epoch_step > self.cfgs.begin_update_lag_epoch:
            self.update_lagrange_multiplier(mean_qr.mean().item() / self.cfgs.cost_scale)

        self.logger.store(
            **{
                'Loss/Loss_actor': policy_loss.item(),
                'Qr/current_Qr': mean_qr[0].mean().item(),
                'Qc/current_Qc': mean_qc[0].mean().item(),
                'Misc/ExplorationNoiseStd': self.actor.std,
                'Misc/lagrange_multiplier': self.lagrangian_multiplier.item(),
            }
        )

    def _soft_update_target(self):
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

        self.logger.log_tabular('Qc/data_Qc')
        self.logger.log_tabular('Qc/target_Qc')
        self.logger.log_tabular('Qc/current_Qc')

        self.logger.log_tabular('Misc/lagrange_multiplier')
        super()._log()
