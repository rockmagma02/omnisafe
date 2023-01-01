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
from omnisafe.common.logger import Logger
from omnisafe.models import ActorBuilder, CriticBuilder
from omnisafe.utils.core import set_optimizer
from omnisafe.utils.offline_dataset import OfflineDataset
from omnisafe.wrappers import wrapper_registry


@registry.register
class CRR:  # pylint: disable=too-many-instance-attributes
    """Implementation of the CRR algorithm.

    References:
        Title: Critic Regularized Regression
        Authors: Ziyu Wang, Alexander Novikov, Konrad Zolna, Jost Tobias Springenberg, Scott Reed,
                Bobak Shahriari, Noah Siegel, Josh Merel, Caglar Gulcehre, Nicolas Heess, Nando de Freitas
        URL:  https://arxiv.org/abs/2006.15134
    """

    def __init__(self, env_id: str, cfgs=None) -> None:
        """Initialize the CRR algorithm.

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

        # setup dataset
        self.dataset = OfflineDataset(cfgs.dataset_path, device=cfgs.device)
        self.obs_mean = self.dataset.obs_mean
        self.obs_std = self.dataset.obs_std

        self.actor.change_device(cfgs.device)
        self.critic.to(cfgs.device)
        self.target_critic.to(cfgs.device)

        what_to_save = {
            'actor': self.actor,
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
        }
        self.logger.setup_torch_saver(what_to_save)
        self.logger.torch_save()

        self.start_time: float
        self.epoch_start_time: float
        self.epoch_step: int

    def learn(self):
        """Learn the model."""
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
        # train critic
        self._update_critic(batch)

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

    def _update_critic(self, batch):  # pylint: disable=too-many-locals
        obs, act, reward, cost, done, next_obs = batch  # pylint: disable=unused-variable

        # compute target
        with torch.no_grad():
            next_action = self.actor.predict(next_obs, deterministic=False)
            qr1_target, qr2_target = self.target_critic(next_obs, next_action)
            qr_target = torch.min(qr1_target, qr2_target)
            qr_target = self.cfgs.reward_scale * reward + self.cfgs.gamma * (1 - done) * qr_target

        qr1, qr2 = self.critic(obs, act)
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

    def _update_actor(self, batch):  # pylint: disable=too-many-locals
        obs, act, reward, cost, done, next_obs = batch  # pylint: disable=unused-variable

        qr1, qr2 = self.critic(obs, act)
        qr_data = torch.min(qr1, qr2)

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
        exp_adv = torch.exp(adv_r.detach() / self.cfgs.beta)

        dist, log_p = self.actor(obs, act)  # pylint: disable=unused-variable
        bc_loss = -log_p
        policy_loss = (exp_adv * bc_loss).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.logger.store(
            **{
                'Loss/Loss_actor': policy_loss.item(),
                'Qr/current_Qr': mean_qr[0].mean().item(),
                'Misc/ExplorationNoiseStd': self.actor.std,
            }
        )

    def _soft_update_target(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.cfgs.smooth_coeff * param.data
                + (1 - self.cfgs.smooth_coeff) * target_param.data
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
        self.actor.change_device(device)
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
        act = self.actor.predict(obs, deterministic=False).detach().numpy()
        return act

    def _log(self):
        self.logger.log_tabular('Epoch')
        self.logger.log_tabular('GradStep')
        self.logger.log_tabular('Time/Total')
        self.logger.log_tabular('Time/Epoch')
        self.logger.log_tabular('Time/Eval')

        self.logger.log_tabular('Loss/Loss_actor')
        self.logger.log_tabular('Loss/Loss_critic')

        self.logger.log_tabular('Qr/current_Qr')
        self.logger.log_tabular('Qr/data_Qr')
        self.logger.log_tabular('Qr/target_Qr')

        self.logger.log_tabular('Metrics/EpCost')
        self.logger.log_tabular('Metrics/EpRet')

        self.logger.log_tabular('Misc/ExplorationNoiseStd')

        self.logger.dump_tabular()
