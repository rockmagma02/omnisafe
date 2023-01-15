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
"""Implementation of the pru algorithm."""

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
from omnisafe.utils.offline_dataset import OfflineDatasetWithSafeFlag as OfflineDataset
from omnisafe.utils.schedule import ConstantSchedule, PiecewiseSchedule
from omnisafe.wrappers import wrapper_registry


@registry.register
class PRUSafe:
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
        self.vae = VAE(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            hidden_sizes=cfgs.model_cfgs.vae_cfgs.hidden_sizes,
            latent_dim=cfgs.model_cfgs.vae_cfgs.latent_dim,
            activation=cfgs.model_cfgs.vae_cfgs.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode,
        )
        self.vae_optimizer = set_optimizer('Adam', module=self.vae, learning_rate=cfgs.vae_lr)

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
        # self.cost_critic = builder.build_critic('q', num_critics=2)
        # self.target_cost_critic = deepcopy(self.cost_critic)
        # self.cost_critic_optimizer = set_optimizer(
        #     'Adam', module=self.cost_critic, learning_rate=cfgs.critic_lr
        # )
        # self.value = builder.build_critic('v')
        # self.target_value = deepcopy(self.value)
        # self.value_optimizer = set_optimizer(
        #     'Adam', module=self.value, learning_rate=cfgs.critic_lr
        # )

        self.beta_in = ConstantSchedule(cfgs.beta_in)
        self.beta_out = PiecewiseSchedule(
            [(cfgs.time_step[0], cfgs.beta_out[0]),
            (cfgs.time_step[1], cfgs.beta_out[1]),
            (cfgs.time_step[2], cfgs.beta_out[2])],
            outside_value=cfgs.beta_out[2],
        )

        alpha_init = torch.as_tensor(cfgs.alpha_init, dtype=torch.float32, device=self.cfgs.device)
        self.log_alpha = nn.Parameter(torch.log(alpha_init), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfgs.alpha_lr)
        self.target_entropy = -torch.prod(torch.as_tensor(self.env.action_space.shape, dtype=torch.float32)).item()

        # setup dataset
        self.dataset = OfflineDataset(cfgs.dataset_path, device=cfgs.device)
        self.obs_mean = self.dataset.obs_mean
        self.obs_std = self.dataset.obs_std

        self.actor.change_device(cfgs.device)
        self.critic.to(cfgs.device)
        self.target_critic.to(cfgs.device)
        # self.cost_critic.to(cfgs.device)
        # self.target_cost_critic.to(cfgs.device)
        self.vae.to(cfgs.device)

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
        vae_loader = self.dataset.get_loader(self.cfgs.batch_size * 20, infinite=False)
        loader = self.dataset.get_loader(self.cfgs.batch_size)

        self._train_vae(vae_loader)

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

    def _train_vae(self, loader):
        """Train the VAE."""
        self.logger.log('Start training VAE...')

        for _ in range(self.cfgs.vae_epochs):
            for grad_step, batch in enumerate(loader):
                obs, act, reward, cost, done, next_obs, is_safe = batch  # pylint: disable=unused-variable
                recon_loss, kl_loss = self.vae.loss(obs, act)
                loss = recon_loss + 1.5 * kl_loss
                self.vae_optimizer.zero_grad()
                loss.backward()
                self.vae_optimizer.step()
                self.logger.log(f'Iteraion: {grad_step}, VAE loss: {loss.item():.4f}, KL loss: {kl_loss.item():.4f}, Recon loss: {recon_loss.item():.4f}')

        self.logger.log('Finish training VAE...')

    def _train(self, grad_step, batch):
        obs, act, reward, cost, done, next_obs, is_safe = batch
        batch_size = obs.shape[0]

        act_sample, logp_sample = self.actor.predict(obs, need_log_prob=True)

        alpha_loss = - (self.log_alpha * (logp_sample + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        act_next = self.actor.predict(next_obs)

        qr1, qr2 = self.critic(obs, act)
        qr = torch.min(qr1, qr2)
        qr1_next, qr2_next = self.target_critic(next_obs, act_next)
        uncertainty = self.uncertainty(obs, act)
        uncertainty_next = self.uncertainty(next_obs, act_next)

        rand_action = (
            torch.FloatTensor(batch_size * self.cfgs.num_sample, act.shape[1])
            .uniform_(-1, 1)
            .to(self.cfgs.device)
        )
        obs_repeat = obs.unsqueeze(1).repeat(1, self.cfgs.num_sample, 1).view(batch_size * self.cfgs.num_sample, obs.shape[1])
        next_obs_repeat = next_obs.unsqueeze(1).repeat(1, self.cfgs.num_sample, 1).view(batch_size * self.cfgs.num_sample, next_obs.shape[1])
        act_curr = self.actor.predict(obs_repeat)
        act_next = self.actor.predict(next_obs_repeat)

        qr1_rand, qr2_rand = self.critic(obs_repeat, rand_action)
        qr1_ood_curr, qr2_ood_curr = self.critic(obs_repeat, act_curr)
        qr1_ood_next, qr2_ood_next = self.critic(next_obs_repeat, act_next)
        uncertainty_rand = self.uncertainty(obs_repeat, rand_action)
        uncertainty_ood_curr = self.uncertainty(obs_repeat, act_curr)
        uncertainty_ood_next = self.uncertainty(next_obs_repeat, act_next)

        beta_in = self.beta_in.value(self.epoch_step * self.cfgs.grad_steps_per_epoch + grad_step)
        beta_out = self.beta_out.value(self.epoch_step * self.cfgs.grad_steps_per_epoch + grad_step)

        qr_next = torch.min(qr1_next, qr2_next)
        qr_target = (reward + (1 - done) * self.cfgs.gamma * (qr_next - beta_in * uncertainty_next)) * (1 - cost) \
                    + cost * (self.cfgs.re)
        qr_target = qr_target.detach()

        critic_loss_in = nn.functional.mse_loss(qr1, qr_target) + nn.functional.mse_loss(qr2, qr_target)

        qr_ood_curr = torch.min(qr1_ood_curr, qr2_ood_curr)
        qr_ood_next = torch.min(qr1_ood_next, qr2_ood_next)

        qr_target_ood = torch.cat(
            [
                torch.max(qr_ood_curr - beta_out * uncertainty_ood_curr, torch.zeros_like(qr_ood_curr)),
                torch.max(qr_ood_next - 0.0001 * uncertainty_ood_next, torch.zeros_like(qr_ood_next)),
            ], dim=0
        )
        qr_target_ood = qr_target_ood.detach()

        qr1_ood = torch.cat([qr1_ood_curr, qr1_ood_next], dim=0)
        qr2_ood = torch.cat([qr2_ood_curr, qr2_ood_next], dim=0)

        critic_loss_out = nn.functional.mse_loss(qr1_ood, qr_target_ood) + nn.functional.mse_loss(qr2_ood, qr_target_ood)

        critic_loss = critic_loss_in + critic_loss_out
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        qr1_sample, qr2_sample = self.critic(obs, act_sample)
        qr_sample = torch.min(qr1_sample, qr2_sample)
        actor_loss = (alpha * logp_sample - qr_sample).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

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
                    # 'Qr/data-rand': (qr1[0].mean() - qr1_rand[0].mean()).item(),
                    # 'Qr/data-curr': (qr1[0].mean() - qr_sample[0].mean()).item(),

                    'Uncertainty/data_uncertainty': uncertainty[0].mean().item(),
                    'Uncertainty/target_uncertainty': uncertainty_next[0].mean().item(),
                    'Uncertainty/curr_uncertainty': uncertainty_ood_curr[0].mean().item(),
                    'Uncertainty/rand_uncertainty': uncertainty_rand[0].mean().item(),
                    # 'Uncertainty/data-rand': (uncertainty[0].mean() - uncertainty_rand[0].mean()).item(),
                    # 'Uncertainty/data-curr': (uncertainty[0].mean() - uncertainty_ood_curr[0].mean()).item(),

                    'Loss/critic_loss': critic_loss.item(),
                    'Loss/critic_loss_in': critic_loss_in.item(),
                    'Loss/critic_loss_out': critic_loss_out.item(),
                    'Loss/actor_loss': actor_loss.item(),
                    'Loss/alpha_loss': alpha_loss.item(),

                    'Misc/alpha': alpha.item(),
                    'Misc/beta_in': beta_in,
                    'Misc/beta_out': beta_out,
                    'Misc/safe_rate': (is_safe.sum() / is_safe.shape[0]).item(),
                }
            )

    def uncertainty(self, obs, act):
        dist = self.vae.encode(obs, act)
        kl = torch.distributions.kl.kl_divergence(dist, torch.distributions.Normal(0, 1)).sum(dim=1)
        return kl * 1e6

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
        # self.logger.log_tabular('Qr/data-rand')
        # self.logger.log_tabular('Qr/data-curr')

        self.logger.log_tabular('Uncertainty/data_uncertainty')
        self.logger.log_tabular('Uncertainty/target_uncertainty')
        self.logger.log_tabular('Uncertainty/curr_uncertainty')
        self.logger.log_tabular('Uncertainty/rand_uncertainty')
        # self.logger.log_tabular('Uncertainty/data-rand')
        # self.logger.log_tabular('Uncertainty/data-curr')

        self.logger.log_tabular('Loss/critic_loss')
        self.logger.log_tabular('Loss/critic_loss_in')
        self.logger.log_tabular('Loss/critic_loss_out')
        self.logger.log_tabular('Loss/actor_loss')
        self.logger.log_tabular('Loss/alpha_loss')

        self.logger.log_tabular('Misc/alpha')
        self.logger.log_tabular('Misc/beta_in')
        self.logger.log_tabular('Misc/beta_out')
        self.logger.log_tabular('Misc/safe_rate')

        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCost')

        self.logger.dump_tabular()
