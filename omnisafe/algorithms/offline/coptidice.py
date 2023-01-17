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
from omnisafe.algorithms.offline.model import ObsDecoder
from omnisafe.common.logger import Logger
from omnisafe.models import ActorBuilder
from omnisafe.utils.core import set_optimizer
from omnisafe.utils.config_utils import namedtuple2dict
from omnisafe.utils.offline_dataset import OfflineDatasetWithInitObs
from omnisafe.wrappers import wrapper_registry


# pylint: disable=invalid-name


@registry.register
class COptiDICE:  # pylint: disable=too-many-instance-attributes
    """Implementation of the COptiDICE algorithm.

    References:
        Title: COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation
        Authors: Jongmin Lee, Cosmin Paduraru, Daniel J. Mankowitz, Nicolas Heess,
                    Doina Precup, Kee-Eung Kim, Arthur Guez
        URL:  https://arxiv.org/abs/2204.08957
    """

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

        # setup model
        buidler = ActorBuilder(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            hidden_sizes=cfgs.model_cfgs.actor_cfgs.hidden_sizes,
            activation=cfgs.model_cfgs.actor_cfgs.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode,
        )
        self.actor = buidler.build_actor(
            actor_type='gaussian_learning',
            act_min=torch.as_tensor(self.env.action_space.low, dtype=torch.float32),
            act_max=torch.as_tensor(self.env.action_space.high, dtype=torch.float32),
        )
        self.actor_optimizer = set_optimizer('Adam', module=self.actor, learning_rate=cfgs.actor_lr)

        self.nu_net = ObsDecoder(
            obs_dim=self.env.observation_space.shape[0],
            out_dim=1,
            hidden_sizes=cfgs.model_cfgs.nu_net_cfgs.hidden_sizes,
            activation=cfgs.model_cfgs.nu_net_cfgs.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode,
        )
        self.nu_net_optimizer = set_optimizer(
            'Adam', module=self.nu_net, learning_rate=cfgs.nu_net_lr
        )

        self.chi_net = ObsDecoder(
            obs_dim=self.env.observation_space.shape[0],
            out_dim=1,
            hidden_sizes=cfgs.model_cfgs.chi_net_cfgs.hidden_sizes,
            activation=cfgs.model_cfgs.chi_net_cfgs.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode,
        )
        self.chi_net_optimizer = set_optimizer(
            'Adam', module=self.chi_net, learning_rate=cfgs.chi_net_lr
        )

        lamb_init = torch.as_tensor(cfgs.lamb_init, dtype=torch.float32, device=cfgs.device)
        self.lamb = nn.Parameter(torch.clamp(lamb_init, 0, 1e3), requires_grad=True)
        self.lamb_optimizer = getattr(torch.optim, cfgs.lamb_optimizer)(
            params=[self.lamb], lr=cfgs.lamb_lr
        )

        tau_init = torch.as_tensor(cfgs.tau_init, dtype=torch.float32, device=cfgs.device)
        self.tau = nn.Parameter(tau_init + 1e-6, requires_grad=True)
        self.tau_optimizer = getattr(torch.optim, cfgs.tau_optimizer)(
            params=[self.tau], lr=cfgs.tau_lr
        )

        # setup dataset
        self.dataset = OfflineDatasetWithInitObs(cfgs.dataset_path, device=cfgs.device)
        self.obs_mean = self.dataset.obs_mean
        self.obs_std = self.dataset.obs_std

        self.actor.change_device(cfgs.device)
        self.nu_net.to(cfgs.device)
        self.chi_net.to(cfgs.device)

        self.fn, self.fn_inv = self._get_f_divergence_fn(cfgs.fn_type)

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
                self.logger.torch_save(itr=epoch + 1)

        self.logger.log('Finish learning...')
        self.logger.close()

    def _train(self, grad_step, batch):
        # train nu, chi, lamb, tau
        self._update_net(batch)

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

    # pylint: disable=too-many-locals
    def _update_net(self, batch):
        obs, act, reward, cost, done, next_obs, init_obs = batch  # pylint: disable=unused-variable
        batch_size = obs.shape[0]

        nu = self.nu_net(obs)
        nu_next = self.nu_net(next_obs)
        adv = self._advantage(reward, cost, done, nu, nu_next)
        w_sa = self._w_sa(adv)

        nu_loss = (
            (1 - self.cfgs.gamma) * self.nu_net(init_obs).mean()
            - self.cfgs.alpha * self.fn(w_sa).mean()
            + (w_sa * adv).mean()
        )

        chi = self.chi_net(obs)  # (batch_size, )
        chi_next = self.chi_net(next_obs)  # (batch_size, )
        chi_init = self.chi_net(init_obs)  # (batch_size, )
        w_sa_no_grad = w_sa.detach()

        ell = (1 - self.cfgs.gamma) * chi_init + w_sa_no_grad * (
            cost + self.cfgs.gamma * (1 - done) * chi_next - chi
        )
        logist = ell / self.tau.item()
        weights = nn.functional.softmax(logist, dim=0) * batch_size
        log_weights = nn.functional.log_softmax(logist, dim=0) + torch.log(
            torch.as_tensor(batch_size, device=self.cfgs.device)
        )
        kl_divergence = (weights * log_weights - weights + 1).mean()
        # cost_ub = (weights * w_sa_no_grad * cost).mean()
        cost_ub = (w_sa_no_grad * cost).mean()
        chi_loss = (weights * ell).mean()
        tau_loss = -self.tau * (kl_divergence.detach() - self.cfgs.cost_ub_eps)

        lamb_loss = -(self.lamb * (cost_ub.detach() - self.cfgs.cost_limit)).mean()

        self.nu_net_optimizer.zero_grad()
        nu_loss.backward()
        self.nu_net_optimizer.step()

        self.chi_net_optimizer.zero_grad()
        chi_loss.backward()
        self.chi_net_optimizer.step()

        self.lamb_optimizer.zero_grad()
        lamb_loss.backward()
        self.lamb_optimizer.step()
        self.lamb.data.clamp_(min=0, max=1e3)

        self.tau_optimizer.zero_grad()
        tau_loss.backward()
        self.tau_optimizer.step()
        self.tau.data.clamp_(min=1e-6)

        self.logger.store(
            **{
                'Loss/Loss_Nu': nu_loss.item(),
                'Loss/Loss_Chi': chi_loss.item(),
                'Loss/Loss_Lamb': lamb_loss.item(),
                'Loss/Loss_Tau': tau_loss.item(),
                'Misc/CostUB': cost_ub.item(),
                'Misc/KL_divergence': kl_divergence.item(),
                'Misc/tau': self.tau.item(),
                'Misc/lagrange_multiplier': self.lamb.item(),
            }
        )

    def _update_actor(self, batch):  # pylint: disable=too-many-locals
        obs, act, reward, cost, done, next_obs, init_obs = batch  # pylint: disable=unused-variable
        dist, log_p = self.actor(obs, act)  # pylint: disable=unused-variable

        nu = self.nu_net(obs)
        nu_next = self.nu_net(next_obs)
        adv = self._advantage(reward, cost, done, nu, nu_next)
        w_sa = self._w_sa(adv)

        policy_loss = -(w_sa * log_p).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.logger.store(
            **{
                'Loss/Loss_Policy': policy_loss.item(),
                'Misc/ExplorationNoiseStd': self.actor.std,
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

        self.logger.log_tabular('Loss/Loss_Nu')
        self.logger.log_tabular('Loss/Loss_Chi')
        self.logger.log_tabular('Loss/Loss_Lamb')
        self.logger.log_tabular('Loss/Loss_Tau')
        self.logger.log_tabular('Loss/Loss_Policy')

        self.logger.log_tabular('Misc/tau')
        self.logger.log_tabular('Misc/lagrange_multiplier')
        self.logger.log_tabular('Misc/CostUB')
        self.logger.log_tabular('Misc/KL_divergence')
        self.logger.log_tabular('Misc/ExplorationNoiseStd')

        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCost')

        self.logger.dump_tabular()

    def _advantage(self, rewrad, cost, done, nu, nu_next):  # pylint: disable=too-many-arguments
        return rewrad - self.lamb.item() * cost + (1 - done) * self.cfgs.gamma * nu_next - nu

    def _w_sa(self, adv):
        return nn.functional.relu(self.fn_inv(adv / self.cfgs.alpha))

    @staticmethod
    def _get_f_divergence_fn(fn_type: str):
        if fn_type == 'kl':

            def fn(x):
                return x * torch.log(x + 1e-10)

            def fn_inv(x):
                return torch.exp(x - 1)

        elif fn_type == 'softchi':

            def fn(x):
                return torch.where(x < 1, x * (torch.log(x + 1e-10) - 1) + 1, 0.5 * (x - 1) ** 2)

            def fn_inv(x):
                return torch.where(x < 0, torch.exp(torch.min(x, torch.zeros_like(x))), x + 1)

        elif fn_type == 'chisquare':

            def fn(x):
                return 0.5 * (x - 1) ** 2

            def fn_inv(x):
                return x + 1

        return fn, fn_inv
