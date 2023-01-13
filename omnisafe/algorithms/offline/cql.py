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
"""Implementation of the cql algorithm."""

import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.model import VAE
from omnisafe.common.logger import Logger
from omnisafe.models import ActorBuilder, CriticBuilder
from omnisafe.utils.core import set_optimizer
from omnisafe.utils.config_utils import namedtuple2dict
from omnisafe.utils.offline_dataset import OfflineDataset
from omnisafe.wrappers import wrapper_registry


@registry.register
class CQL:
    def __init__(self, env_id: str, cfgs=None) -> None:
        """Initialize the COptiDICE algorithm.

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

        builder = ActorBuilder(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            hidden_sizes=cfgs.model_cfgs.actor_cfgs.hidden_sizes,
            activation=cfgs.model_cfgs.actor_cfgs.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode,
        )
        self.actor = builder.build_actor(
            actor_type='gaussian_stdnet',
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

        alpha_init = torch.as_tensor(cfgs.alpha_init, dtype=torch.float32, device=self.cfgs.device)
        self.log_alpha = nn.Parameter(torch.log(alpha_init), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfgs.alpha_lr)
        self.target_entropy = -torch.prod(torch.as_tensor(self.env.action_space.shape, dtype=torch.float32)).item()

        self.target_action_gap = self.cfgs.lagrange_thresh
        self.log_alpha_prime = nn.Parameter(torch.zeros(1, device=self.cfgs.device), requires_grad=True)
        self.alpha_prime_optimizer = torch.optim.Adam([self.log_alpha_prime], lr=cfgs.critic_lr)

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

    def learn(self) -> None:
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
                self.logger.torch_save(epoch)

        self.logger.log('Finish learning...')
        self.logger.close()

    def _train(self, grad_step, batch):
        obs, act, reward, cost, done, next_obs = batch
        batch_size = obs.shape[0]

        act_sample, logp_sample = self.actor.predict(obs, need_log_prob=True)

        alpha_loss = - (self.log_alpha * (logp_sample + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp()

        qr1_sample, qr2_sample = self.critic(obs, act_sample)
        qr_sample = torch.min(qr1_sample, qr2_sample)

        if self.epoch_step < self.cfgs.start_train_policy:
            _, data_logp = self.actor(obs, act)
            policy_loss = (alpha * logp_sample - data_logp).mean()
        else:
            policy_loss = (alpha * logp_sample - qr_sample).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        qr1, qr2 = self.critic(obs, act)

        next_act = self.actor.predict(next_obs)
        # curr_act = self.actor.predict(obs)

        qr1_target, qr2_target = self.target_critic(next_obs, next_act)
        qr_target = torch.min(qr1_target, qr2_target)
        qr_target = reward + self.cfgs.gamma * (1 - done) * qr_target
        qr_target = qr_target.detach()

        critic_loss_in = nn.functional.mse_loss(qr1, qr_target) + nn.functional.mse_loss(qr2, qr_target)

        rand_action = (
            torch.FloatTensor(batch_size * self.cfgs.num_sample, act.shape[1])
            .uniform_(-1, 1)
            .to(self.cfgs.device)
        )
        obs_repeat = obs.unsqueeze(1).repeat(1, self.cfgs.num_sample, 1).view(batch_size * self.cfgs.num_sample, obs.shape[1])
        next_obs_repeat = next_obs.unsqueeze(1).repeat(1, self.cfgs.num_sample, 1).view(batch_size * self.cfgs.num_sample, next_obs.shape[1])
        act_curr, logp_curr = self.actor.predict(obs_repeat, need_log_prob=True)
        act_next, logp_next = self.actor.predict(next_obs_repeat, need_log_prob=True)
        qr1_rand, qr2_rand = self.critic(obs_repeat, rand_action)
        qr1_ood_curr, qr2_ood_curr = self.critic(obs_repeat, act_curr)
        qr1_ood_next, qr2_ood_next = self.critic(next_obs_repeat, act_next)
        qr1_rand = qr1_rand.view(batch_size, self.cfgs.num_sample)
        qr2_rand = qr2_rand.view(batch_size, self.cfgs.num_sample)
        qr1_ood_curr = qr1_ood_curr.view(batch_size, self.cfgs.num_sample)
        qr2_ood_curr = qr2_ood_curr.view(batch_size, self.cfgs.num_sample)
        qr1_ood_next = qr1_ood_next.view(batch_size, self.cfgs.num_sample)
        qr2_ood_next = qr2_ood_next.view(batch_size, self.cfgs.num_sample)
        logp_curr = logp_curr.view(batch_size, self.cfgs.num_sample)
        logp_next = logp_next.view(batch_size, self.cfgs.num_sample)

        random_density = torch.log(torch.as_tensor(0.5 ** act.shape[-1], device=self.cfgs.device))
        cat_qr1 = torch.cat(
            [qr1_rand - random_density, qr1_ood_next - logp_next.detach(), qr1_ood_curr - logp_curr.detach()],
            dim=1
        )
        cat_qr2 = torch.cat(
            [qr2_rand - random_density, qr2_ood_next - logp_next.detach(), qr2_ood_curr - logp_curr.detach()],
            dim=1
        )
        min_critic_loss1 = torch.logsumexp(cat_qr1, dim=1).mean() * self.cfgs.min_q_weight
        min_critic_loss2 = torch.logsumexp(cat_qr2, dim=1).mean() * self.cfgs.min_q_weight

        min_critic_loss1 = min_critic_loss1 - qr1.mean() * self.cfgs.min_q_weight
        min_critic_loss2 = min_critic_loss2 - qr2.mean() * self.cfgs.min_q_weight

        alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0, max=1e6)
        min_critic_loss1 = alpha_prime * (min_critic_loss1 - self.target_action_gap)
        min_critic_loss2 = alpha_prime * (min_critic_loss2 - self.target_action_gap)

        self.alpha_prime_optimizer.zero_grad()
        alpha_prime_loss = (-min_critic_loss1 - min_critic_loss2) * 0.5
        alpha_prime_loss.backward(retain_graph=True)
        self.alpha_prime_optimizer.step()

        critic_loss = critic_loss_in + min_critic_loss1 + min_critic_loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        self._soft_update_target()

        if grad_step == self.cfgs.grad_steps_per_epoch - 1:
            self.logger.store(
                **{
                    'Epoch': self.epoch_step + 1,
                    'GradStep': (self.epoch_step + 1) * self.cfgs.grad_steps_per_epoch,
                    'Time/Epoch': time.time() - self.epoch_start_time,
                    'Time/Total': time.time() - self.start_time,

                    'Qr/data_Qr': qr1[0].mean().item(),
                    'Qr/target_Qr': qr_target[0].mean().item(),
                    'Qr/curr_Qr': qr_sample[0].mean().item(),
                    'Qr/rand_Qr': qr1_rand[0].mean().item(),

                    'Loss/actor_loss': policy_loss.item(),
                    'Loss/critic_loss': critic_loss.item(),
                    'Loss/critic_loss_in': critic_loss_in.item(),
                    'Loss/min_critic_loss1': min_critic_loss1.item(),

                    'Misc/alpha_prime': alpha_prime.item(),
                    'Misc/alpha': self.log_alpha.exp().item(),
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

        self.logger.log_tabular('Qr/data_Qr')
        self.logger.log_tabular('Qr/target_Qr')
        self.logger.log_tabular('Qr/curr_Qr')
        self.logger.log_tabular('Qr/rand_Qr')

        self.logger.log_tabular('Loss/actor_loss')
        self.logger.log_tabular('Loss/critic_loss')
        self.logger.log_tabular('Loss/critic_loss_in')
        self.logger.log_tabular('Loss/min_critic_loss1')

        self.logger.log_tabular('Misc/alpha')
        self.logger.log_tabular('Misc/alpha_prime')

        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCost')

        self.logger.dump_tabular()
