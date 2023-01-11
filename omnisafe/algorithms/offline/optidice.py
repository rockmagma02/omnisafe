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
from omnisafe.algorithms.offline.model import ObsDecoder, ObsActDecoder
from omnisafe.common.logger import Logger
from omnisafe.models import ActorBuilder
from omnisafe.utils.core import set_optimizer
from omnisafe.utils.config_utils import namedtuple2dict
from omnisafe.utils.offline_dataset import OfflineDatasetWithInitObs
from omnisafe.wrappers import wrapper_registry


@registry.register
class OptiDICE:
    """Implementation of the COptiDICE algorithm.

    References:
        Title: OptiDICE: Offline Policy Optimization via Stationary Distribution Correction Estimation
        Authors: Lee, JongminJeon, WonseokLee, Byung-JunPineau, JoelleKim, Kee-Eung
        URL:  https://arxiv.org/abs/2106.10783
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
        self.logger = Logger(exp_name=cfgs.exp_name, data_dir=cfgs.data_dir, seed=cfgs.seed)
        self.logger.save_config(namedtuple2dict(cfgs))
        # set seed
        seed = int(cfgs.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.set_seed(seed)

        self.v_net = ObsDecoder(
            obs_dim=self.env.observation_space.shape[0],
            out_dim=1,
            hidden_sizes=cfgs.model_cfgs.v_cfgs.hidden_sizes,
            activation=cfgs.model_cfgs.v_cfgs.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode
        )
        self.v_optimizer = set_optimizer('Adam', module=self.v_net, learning_rate=cfgs.v_lr)

        self.e_net = ObsActDecoder(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            out_dim=1,
            hidden_sizes=cfgs.model_cfgs.e_cfgs.hidden_sizes,
            activation=cfgs.model_cfgs.e_cfgs.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode
        )
        self.e_optimizer = set_optimizer('Adam', module=self.e_net, learning_rate=cfgs.e_lr)

        # GenDICE regularization, i.e., E[w] = 1.
        self.lamb_v = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.cfgs.device), requires_grad=True)
        self.lamb_e = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.cfgs.device), requires_grad=True)
        self.lamb_v_optimizer = torch.optim.Adam([self.lamb_v], lr=cfgs.lamb_v_lr)
        self.lamb_e_optimizer = torch.optim.Adam([self.lamb_e], lr=cfgs.lamb_e_lr)

        if cfgs.fn_type == 'chisquare':
            self._f_fn = lambda x: 0.5 * (x - 1) ** 2
            self._f_prime_inv_fn = lambda x: x + 1
            self._g_fn = lambda x: 0.5 * (nn.functional.relu(x + 1) - 1) ** 2
            self._r_fn = lambda x: nn.functional.relu(self._f_prime_inv_fn(x))
            self._log_r_fn = lambda x: torch.where(x < 0, torch.log(1e-10), torch.log(torch.max(x, torch.zeros_like(x)) + 1))
        elif cfgs.fn_type == 'kl':
            self._f_fn = lambda x: x * torch.log(x + 1e-10)
            self._f_prime_inv_fn = lambda x: torch.exp(x - 1)
            self._g_fn = lambda x: torch.exp(x - 1) * (x - 1)
            self._r_fn = lambda x: self._f_prime_inv_fn(x)
            self._log_r_fn = lambda x: x - 1
        elif cfgs.fn_type == 'elu':
            self._f_fn = lambda x: torch.where(x < 1, x * (torch.log(x + 1e-10) - 1) + 1, 0.5 * (x - 1) ** 2)
            self._f_prime_inv_fn = lambda x: torch.where(x < 0, torch.exp(torch.min(x, torch.zeros_like(x))), x + 1)
            self._g_fn = lambda x: torch.where(x < 0, torch.exp(torch.min(x, torch.zeros_like(x))) * (torch.min(x, torch.zeros_like(x)) - 1) + 1, 0.5 * x ** 2)
            self._r_fn = lambda x: self._f_prime_inv_fn(x)
            self._log_r_fn = lambda x: torch .where(x < 0, x, torch.log(torch.max(x, torch.zeros_like(x)) + 1))
        else:
            raise NotImplementedError()

        # setup actor
        buidler = ActorBuilder(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            hidden_sizes=cfgs.model_cfgs.actor_cfgs.hidden_sizes,
            activation=cfgs.model_cfgs.actor_cfgs.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode,
        )
        self.actor = buidler.build_actor(
            actor_type='gaussian_stdnet',
            act_min=torch.as_tensor(self.env.action_space.low, dtype=torch.float32),
            act_max=torch.as_tensor(self.env.action_space.high, dtype=torch.float32),
        )
        self.actor_optimizer = set_optimizer('Adam', module=self.actor, learning_rate=cfgs.actor_lr)

        self.data_actor = buidler.build_actor(
            actor_type='gaussian_stdnet',
            act_min=torch.as_tensor(self.env.action_space.low, dtype=torch.float32),
            act_max=torch.as_tensor(self.env.action_space.high, dtype=torch.float32),
        )
        self.data_actor_optimizer = set_optimizer('Adam', module=self.actor, learning_rate=cfgs.actor_lr)

        alpha_init = torch.as_tensor(cfgs.alpha_init, dtype=torch.float32, device=self.cfgs.device)
        self.log_alpha = nn.Parameter(torch.log(alpha_init), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfgs.alpha_lr)
        self.target_entropy = -torch.prod(torch.as_tensor(self.env.action_space.shape, dtype=torch.float32)).item()

        # setup dataset
        self.dataset = OfflineDatasetWithInitObs(cfgs.dataset_path, device=cfgs.device)
        self.obs_mean = self.dataset.obs_mean
        self.obs_std = self.dataset.obs_std

        self.actor.change_device(cfgs.device)
        self.data_actor.change_device(cfgs.device)
        self.e_net.to(cfgs.device)
        self.v_net.to(cfgs.device)

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
        obs, act, reward, cost, done, next_obs, init_obs = batch  # pylint: disable=unused-variable

        v_init = self.v_net(init_obs)
        v = self.v_net(obs)
        v_next = self.v_net(next_obs)

        e_v = reward + (1 - done) * self.cfgs.gamma * v_next - v
        preactivation_v = (e_v - self.lamb_v) / self.cfgs.dice_alpha
        w_v = self._r_fn(preactivation_v)
        f_w_v = self._g_fn(preactivation_v)

        e = self.e_net(obs, act)
        preactivation_e = (e - self.lamb_e) / self.cfgs.dice_alpha
        w_e = self._r_fn(preactivation_e)
        f_w_e = self._g_fn(preactivation_e)

        v_loss = self._v_loss(v_init, e_v, w_v, f_w_v)
        lamb_v_loss = self._lamb_v_loss(e_v, w_v, f_w_v)

        e_loss = self._e_loss(e_v.detach(), e, w_e, f_w_e)
        lamb_e_loss = self._lamb_e_loss(e_v.detach(), w_e, f_w_e)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        self.lamb_v_optimizer.zero_grad()
        lamb_v_loss.backward()
        self.lamb_v_optimizer.step()

        self.e_optimizer.zero_grad()
        e_loss.backward()
        self.e_optimizer.step()

        self.lamb_e_optimizer.zero_grad()
        lamb_e_loss.backward()
        self.lamb_e_optimizer.step()

        policy_loss, alpha_loss = self._policy_loss(obs, act, w_e.detach())
        data_policy_loss = self._data_policy_loss(obs, act)

        if self.epoch_step >= self.cfgs.warmup_epochs:
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.data_actor_optimizer.zero_grad()
            data_policy_loss.backward()
            self.data_actor_optimizer.step()

        if grad_step == self.cfgs.grad_steps_per_epoch - 1:
            self.logger.store(
                **{
                    'Epoch': self.epoch_step + 1,
                    'GradStep': (self.epoch_step + 1) * self.cfgs.grad_steps_per_epoch,
                    'Time/Epoch': time.time() - self.epoch_start_time,
                    'Time/Total': time.time() - self.start_time,
                }
            )

    def _v_loss(self, v_init, e_v, w_v, f_w_v):
        v_loss0 = (1 - self.cfgs.gamma) * v_init.mean()
        v_loss1 = (-self.cfgs.dice_alpha * f_w_v).mean()
        v_loss2 = (w_v * (e_v - self.lamb_v)).mean()
        v_loss3 = self.lamb_v
        v_loss = v_loss0 + v_loss1 + v_loss2 + v_loss3

        self.logger.store(
            **{
                'Loss/v': v_loss.item(),
            }
        )

        return v_loss

    def _lamb_v_loss(self, e_v, w_v, f_w_v):
        e_v = e_v.detach()
        w_v = w_v.detach()
        f_w_v = f_w_v.detach()
        lamb_v_loss = (- self.cfgs.dice_alpha * f_w_v + w_v * (e_v - self.lamb_v) + self.lamb_v).mean()

        self.logger.store(
            **{
                'Loss/lamb_v': lamb_v_loss.item(),
                'Misc/lamb_v': self.lamb_v.item(),
            }
        )

        return lamb_v_loss

    def _e_loss(self, e_v, e, w_e, f_w_e):
        e_loss = (self.cfgs.dice_alpha * f_w_e - w_e * (e_v - self.lamb_v)).mean()

        self.logger.store(
            **{
                'Loss/e': e_loss.item(),
            }
        )

        return e_loss

    def _lamb_e_loss(self, e_v, w_e, f_w_e):
        w_e = w_e.detach()
        f_w_e = f_w_e.detach()
        lamb_e_loss = lamb_e_loss = (- self.cfgs.dice_alpha * f_w_e + w_e * (e_v - self.lamb_e) + self.lamb_e).mean()

        self.logger.store(
            **{
                'Loss/lamb_e': lamb_e_loss.item(),
                'Misc/lamb_e': self.lamb_e.item(),
            }
        )

        return lamb_e_loss

    def _policy_loss(self, obs, act, w_e):
        act_smaple, logp_sample = self.actor.predict(obs, need_log_prob=True)
        e_sample = self.e_net(obs, act_smaple)
        sample_log_w_e = self._log_r_fn((e_sample - self.lamb_e.item()) / self.cfgs.dice_alpha)
        _, data_logp_sample = self.data_actor(obs, act_smaple)
        kl = (logp_sample - data_logp_sample)
        policy_loss = (sample_log_w_e - kl).mean()
        alpha = torch.exp(self.log_alpha).item()
        policy_loss += alpha * logp_sample.mean()
        alpha_loss = - (self.log_alpha * (logp_sample.mean().item() + self.target_entropy)).mean()

        self.logger.store(
            **{
                'Loss/policy': policy_loss.item(),
                'Loss/alpha': alpha_loss.item(),
                'Misc/alpha': alpha,
            }
        )

        return policy_loss, alpha_loss

    def _data_policy_loss(self, obs, act):
        _, data_logp = self.data_actor(obs, act)
        data_policy_loss = - data_logp.mean()

        self.logger.store(
            **{
                'Loss/data_policy': data_policy_loss.item(),
            }
        )

        return data_policy_loss

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

        self.logger.log_tabular('Loss/v')
        self.logger.log_tabular('Loss/lamb_v')
        self.logger.log_tabular('Loss/e')
        self.logger.log_tabular('Loss/lamb_e')
        self.logger.log_tabular('Loss/policy')
        self.logger.log_tabular('Loss/alpha')
        self.logger.log_tabular('Loss/data_policy')

        self.logger.log_tabular('Misc/lamb_v')
        self.logger.log_tabular('Misc/lamb_e')
        self.logger.log_tabular('Misc/alpha')

        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCost')

        self.logger.dump_tabular()
