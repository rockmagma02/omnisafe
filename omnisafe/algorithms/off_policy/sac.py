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
"""Implementation of the SAC algorithm."""

import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from omnisafe.algorithms import registry
from omnisafe.common.base_buffer import BaseBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils import core, distributed_utils
from omnisafe.utils.config_utils import namedtuple2dict
from omnisafe.wrappers import wrapper_registry


# pylint: disable=invalid-name


@registry.register
class SAC:  # pylint: disable=too-many-instance-attributes
    """The Soft Actor-Critic (SAC) algorithm.

    References:
        Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
        URL: https://arxiv.org/abs/1801.01290
    """

    def __init__(self, env_id: str, cfgs=None) -> None:
        self.env_id = env_id
        self.algo = self.__class__.__name__
        self.cfgs = deepcopy(cfgs)
        self.env = wrapper_registry.get(self.cfgs.wrapper_type)(
            env_id,
            use_cost=self.cfgs.use_cost,
            max_ep_len=self.cfgs.max_ep_len,
        )

        seed = cfgs.seed + 10000 * distributed_utils.proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.set_seed(seed)

        self.logger = Logger(exp_name=cfgs.exp_name, data_dir=cfgs.data_dir, seed=cfgs.seed)
        self.logger.save_config(namedtuple2dict(cfgs))

        self.actor_critic = ConstraintActorQCritic(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            standardized_obs=self.cfgs.standardized_obs,
            model_cfgs=self.cfgs.model_cfgs,
        )

        self._init_mpi()

        self.buf = BaseBuffer(
            obs_dim=self.env.observation_space.shape,
            act_dim=self.env.action_space.shape,
            size=cfgs.replay_buffer_cfgs.size,
            batch_size=cfgs.replay_buffer_cfgs.batch_size,
        )
        self.actor_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.actor, learning_rate=cfgs.actor_lr
        )
        self.critic_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.critic, learning_rate=cfgs.critic_lr
        )
        if cfgs.use_cost:
            self.cost_critic_optimizer = core.set_optimizer(
                'Adam', module=self.actor_critic.cost_critic, learning_rate=cfgs.critic_lr
            )

        if cfgs.use_adjustable_temperature:
            self.log_alpha = torch.tensor(np.log(cfgs.init_temperature), requires_grad=True)
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape)).item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfgs.alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(cfgs.init_temperature))

        self.scheduler = self.set_learning_rate_scheduler()
        self._ac_training_setup()

        what_to_save = {
            'pi': self.actor_critic.actor,
            'obs_oms': self.actor_critic.obs_oms,
        }
        self.logger.setup_torch_saver(what_to_save)
        self.logger.torch_save()

        self.start_time: float

    def learn(self):
        """Learn the policy."""
        self.logger.log('Start learning')
        self.start_time = time.time()

        for steps in range(0, self.cfgs.steps_per_epoch * self.cfgs.epochs, self.cfgs.update_every):
            use_rand_action = steps < self.cfgs.start_steps
            self.env.roll_out(
                self.actor_critic,
                self.buf,
                self.logger,
                deterministic=False,
                use_rand_action=use_rand_action,
                ep_steps=self.cfgs.update_every,
            )

            if steps >= self.cfgs.update_after:
                for _ in range(self.cfgs.update_every):
                    batch = self.buf.sample_batch()
                    self.update(batch)

            if steps % self.cfgs.steps_per_epoch == 0 and steps > 0:
                epoch = steps // self.cfgs.steps_per_epoch
                if epoch % self.cfgs.save_freq == 0:
                    self.logger.torch_save(itr=epoch)

                self.test_agent()

                # if self.scheduler and self.cfgs.linear_lr_decay:
                #     current_lr = self.scheduler.get_last_lr()[0]
                #     self.scheduler.step()
                # else:
                #     current_lr = self.cfgs.actor_lr

                self.log(epoch, steps)

    def update(self, batch):
        """Update the policy."""
        self.update_value_net(batch)

        if self.cfgs.use_cost:
            self.update_cost_net(batch)

        self.freeze_q()
        self.update_policy(batch)
        self.unfreeze_q()

        self.polyak_update_target()

    def update_value_net(self, batch):
        """Update the value network."""
        self.critic_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(batch)
        loss_q.backward()
        self.critic_optimizer.step()
        self.logger.store(**{'Loss/Value': loss_q.item(), 'QVals': q_info['QVals']})

    def compute_loss_q(self, batch):
        """Compute the loss for the value network."""
        obs, act, rew, obs_next, done = (
            batch['obs'],
            batch['act'],
            batch['rew'],
            batch['obs_next'],
            batch['done'],
        )

        alpha = self.log_alpha.exp()

        q1, q2 = self.actor_critic.critic(obs, act)
        with torch.no_grad():
            next_act, logp_next = self.ac_targ.actor.predict(obs_next, need_log_prob=True)
            q_target = torch.min(*self.ac_targ.critic(obs_next, next_act))
            q_target = rew + self.cfgs.gamma * (1 - done) * (q_target - alpha * logp_next)

        loss_q = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(q2, q_target)
        q_info = dict(QVals=q1.mean().item())
        return loss_q, q_info

    def update_cost_net(self, batch):
        """Update the cost network."""
        self.cost_critic_optimizer.zero_grad()
        loss_cost, cost_info = self.compute_loss_cost(batch)
        loss_cost.backward()
        self.cost_critic_optimizer.step()
        self.logger.store(**{'Loss/Cost': loss_cost.item(), 'QCosts': cost_info['QCosts']})

    def compute_loss_cost(self, batch):
        """Compute the loss for the cost network."""
        obs, act, cost, obs_next, done = (
            batch['obs'],
            batch['act'],
            batch['cost'],
            batch['obs_next'],
            batch['done'],
        )

        alpha = self.log_alpha.exp()

        q1, q2 = self.actor_critic.cost_critic(obs, act)
        with torch.no_grad():
            next_act, logp_next = self.ac_targ.actor.predict(obs_next, need_log_prob=True)
            q_target = torch.min(*self.ac_targ.cost_critic(obs_next, next_act))
            q_target = cost + self.cfgs.gamma * (1 - done) * (q_target - alpha * logp_next)

        loss_q = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(q2, q_target)
        q_info = dict(QVals=q1.mean().item())
        return loss_q, q_info

    def freeze_q(self):
        """Freeze Q."""
        for p in self.actor_critic.critic.parameters():
            p.requires_grad = False

        if self.cfgs.use_cost:
            for p in self.actor_critic.cost_critic.parameters():
                p.requires_grad = False

    def unfreeze_q(self):
        """Unfreeze Q."""
        for p in self.actor_critic.critic.parameters():
            p.requires_grad = True

        if self.cfgs.use_cost:
            for p in self.actor_critic.cost_critic.parameters():
                p.requires_grad = True

    def update_policy(self, batch):
        """Update policy."""
        loss_pi, loss_alpha, pi_info = self.compute_loss_pi(batch)

        self.actor_optimizer.zero_grad()
        loss_pi.backward()
        self.actor_optimizer.step()
        self.logger.store(**{'Loss/Policy': loss_pi.item(), **pi_info})

        if self.cfgs.use_adjustable_temperature:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
            self.logger.store(**{'Loss/Alpha': loss_alpha.item()})

    def compute_loss_pi(self, batch):
        """Compute loss for policy."""
        alpha = self.log_alpha.exp()
        obs = batch['obs']
        act, logp = self.actor_critic.actor.predict(obs, need_log_prob=True)
        q1, q2 = self.actor_critic.critic(obs, act)
        q = torch.min(q1, q2)
        loss_pi = (alpha * logp - q).mean()

        alpha = self.log_alpha.exp()
        loss_alpha = -(alpha * (logp.mean().item() + self.target_entropy)).mean()
        pi_info = dict(LogPi=logp.mean().item(), Alpha=alpha.item())
        return loss_pi, loss_alpha, pi_info

    def polyak_update_target(self):
        """Polyak update target network."""
        with torch.no_grad():
            for param, param_targ in zip(self.actor_critic.parameters(), self.ac_targ.parameters()):
                # Notes: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                param_targ.data.mul_(self.cfgs.polyak)
                param_targ.data.add_((1 - self.cfgs.polyak) * param.data)

    def test_agent(self):
        """Test agent."""
        for _ in range(self.cfgs.num_test_episodes):
            # self.env.set_rollout_cfgs(deterministic=True, rand_a=False)
            self.env.reset()
            self.env.roll_out(
                self.actor_critic,
                self.buf,
                self.logger,
                deterministic=False,
                use_rand_action=False,
                ep_steps=self.env.max_ep_len,
                test=True,
            )

    def _init_mpi(self):
        """Initialize MPI specifics."""

        if distributed_utils.num_procs() > 1:
            # Avoid slowdowns from PyTorch + MPI combo
            distributed_utils.setup_torch_for_mpi()
            start = time.time()
            self.logger.log('INFO: Sync actor critic parameters')
            # Sync params across cores: only once necessary, grads are averaged!
            distributed_utils.sync_params(self.actor_critic)
            self.logger.log(f'Done! (took {time.time()-start:0.3f} sec.)')

    def set_learning_rate_scheduler(self):
        """Set up learning rate scheduler."""

        scheduler = None
        if self.cfgs.linear_lr_decay:
            # Linear anneal
            def linear_anneal(epoch):
                return 1 - epoch / self.cfgs.epochs

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.actor_optimizer, lr_lambda=linear_anneal
            )
        return scheduler

    def _ac_training_setup(self):
        """Set up target network for off_policy training."""
        self.ac_targ = deepcopy(self.actor_critic)
        # Freeze target networks with respect to optimizer (only update via polyak averaging)
        for param in self.ac_targ.actor.parameters():
            param.requires_grad = False
        for param in self.ac_targ.critic.parameters():
            param.requires_grad = False
        for param in self.ac_targ.cost_critic.parameters():
            param.requires_grad = False

    def log(self, epoch, total_steps):
        """Log info about epoch."""
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCost')
        self.logger.log_tabular('Metrics/EpLen')
        self.logger.log_tabular('Test/EpRet')
        self.logger.log_tabular('Test/EpCost')
        self.logger.log_tabular('Test/EpLen')
        self.logger.log_tabular('Values/V')
        self.logger.log_tabular('Loss/Value')
        self.logger.log_tabular('QVals')
        self.logger.log_tabular('Loss/Policy')
        self.logger.log_tabular('LogPi')
        self.logger.log_tabular('Loss/Alpha')
        self.logger.log_tabular('Alpha')
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.log_tabular('TotalEnvSteps', total_steps)

        self.logger.dump_tabular()
