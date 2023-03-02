# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of the Policy Gradient algorithm."""

import time
from copy import deepcopy
from typing import Tuple, Union

import torch
from torch.nn import functional as F

from omnisafe.adapter import OffPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils import distributed


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class DDPG(BaseAlgo):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:

        - Title: Continuous control with deep reinforcement learning
        - Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess,
        Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.
        - URL: `DDPG <https://arxiv.org/abs/1509.02971>`_
    """

    def _init_env(self) -> None:
        self._env = OffPolicyAdapter(self._env_id, self._cfgs.num_envs, self._seed, self._cfgs)
        assert self._cfgs.steps_per_epoch % (distributed.world_size() * self._cfgs.num_envs) == 0, (
            'The number of steps per epoch is not divisible by the number of ' 'environments.'
        )
        self._total_steps = (
            self._cfgs.total_steps // distributed.world_size() // self._cfgs.num_envs
        )
        self._steps_per_epoch = (
            self._cfgs.steps_per_epoch // distributed.world_size() // self._cfgs.num_envs
        )
        self._steps_per_sample = self._cfgs.steps_per_sample
        assert self._steps_per_epoch % self._steps_per_sample == 0, (
            'The number of steps per epoch is not divisible by the number of ' 'steps per sample.'
        )

    def _init_model(self) -> None:
        self._actor_critic = ConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

    def _init(self) -> None:
        self._buf = VectorOffPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._cfgs.replay_buffer_cfgs.size,
            batch_size=self._cfgs.replay_buffer_cfgs.batch_size,
            num_envs=self._cfgs.num_envs,
            device=self._device,
        )
        self._target_actor_critic = deepcopy(self._actor_critic)
        # freeze target networks with respect to optimizer (only update via polyak averaging)
        for param in self._target_actor_critic.actor.parameters():
            param.requires_grad = False
        for param in self._target_actor_critic.reward_critic.parameters():
            param.requires_grad = False
        for param in self._target_actor_critic.cost_critic.parameters():
            param.requires_grad = False

    def _init_log(self) -> None:
        self._logger = Logger(
            output_dir=self._cfgs.data_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.use_tensorboard,
            use_wandb=self._cfgs.use_wandb,
            config=self._cfgs,
        )
        if self._cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save = {
                'pi': self._actor_critic.actor,
                'obs_normalizer': obs_normalizer,
            }
        else:
            what_to_save = {'pi': self._actor_critic.actor}

        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key('Metrics/EpRet', window_length=50)
        self._logger.register_key('Metrics/EpCost', window_length=50)
        self._logger.register_key('Metrics/EpLen', window_length=50)

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/LR')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward_critic1')

        if self._cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost')
        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

    def learn(self) -> Tuple[Union[int, float], ...]:
        """This is main function for algorithm update, divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.
        """
        self._logger.log('INFO: Start training')
        start_time = time.time()
        step = 0
        while step < int(self._total_steps):
            roll_out_time = 0.0
            epoch_time = time.time()
            samples_per_epoch = self._steps_per_epoch // self._steps_per_sample
            # Collect data from environment
            for i in range(samples_per_epoch):
                roll_out_start = time.time()
                self._env.roll_out(
                    steps_per_sample=self._steps_per_sample,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                    epoch_end=i == samples_per_epoch - 1,
                    use_rand_action=step <= self._cfgs.random_steps,
                )
                roll_out_end = time.time()
                roll_out_time += roll_out_end - roll_out_start
                step += self._steps_per_sample

                # Update parameters
                if step > self._cfgs.start_learning_steps:
                    for j in range(self._steps_per_sample):
                        if j % self._cfgs.update_cycle == 0:
                            update_counts = i // self._cfgs.update_cycle
                            # In TD3, we update the actor and critic networks separately
                            self._update(update_policy=update_counts % self._cfgs.policy_delay == 0)
                # If we haven't updated the network, log 0 for the loss
                else:
                    self._log_zero()

            # Count the number of epoch
            epoch = step // self._steps_per_epoch
            self._logger.store(**{'Time/Rollout': roll_out_time})
            self._logger.store(**{'Time/Update': time.time() - roll_out_time - epoch_time})

            if step > self._cfgs.start_learning_steps:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                **{
                    'TotalEnvSteps': step,
                    'Time/FPS': self._cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': self._actor_critic.actor_scheduler.get_last_lr()[0],
                }
            )

            self._logger.dump_tabular()

            # save model to disk
            if (step // self._steps_per_epoch + 1) % self._cfgs.save_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        return ep_ret, ep_cost, ep_len

    def _update(self, update_policy: bool) -> None:
        data = self._buf.sample_batch()
        obs, act, reward, cost, done, next_obs = (
            data['obs'],
            data['act'],
            data['reward'],
            data['cost'],
            data['done'],
            data['next_obs'],
        )
        self._update_rewrad_critic(obs, act, reward, done, next_obs)
        if self._cfgs.use_cost:
            self._update_cost_critic(obs, act, cost, done, next_obs)
        if update_policy:
            self._update_actor(obs)
        self._polyak_update()

    def _update_rewrad_critic(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_action = self._target_actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_value = self._target_actor_critic.reward_critic(next_obs, next_action)[0]
            target_q_value = reward + self._cfgs.gamma * (1 - done) * next_q_value
        q_value = self._actor_critic.reward_critic(obs, act)[0]
        loss = F.mse_loss(q_value, target_q_value)

        if self._cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.critic_norm_coeff
        self._logger.store(
            **{
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic1': q_value.mean().item(),
            }
        )
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(), self._cfgs.max_grad_norm
            )
        distributed.avg_grads(self._actor_critic.reward_critic)
        self._actor_critic.reward_critic_optimizer.step()

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_action = self._target_actor_critic.actor.predict(next_obs, deterministic=True)
            next_qc_value = self._target_actor_critic.cost_critic(next_obs, next_action)[0]
            target_qc_value = cost + self._cfgs.gamma * (1 - done) * next_qc_value
        qc_value = self._actor_critic.cost_critic(obs, act)[0]
        loss = F.mse_loss(qc_value, target_qc_value)

        if self._cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(), self._cfgs.max_grad_norm
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store(
            **{
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/cost': qc_value.mean().item(),
            }
        )

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
    ) -> None:
        loss = self._loss_pi(obs)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.actor.parameters(), self._cfgs.max_grad_norm
            )
        self._actor_critic.actor_optimizer.step()
        self._logger.store(
            **{
                'Loss/Loss_pi': loss.mean().item(),
            }
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        action = self._actor_critic.actor.predict(obs, deterministic=True)
        loss = -self._actor_critic.reward_critic(obs, action)[0].mean()
        return loss

    def _polyak_update(self) -> None:
        for target_param, param in zip(
            self._target_actor_critic.parameters(),
            self._actor_critic.parameters(),
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self._cfgs.polyak) + param.data * self._cfgs.polyak
            )

    def _log_zero(self) -> None:
        self._logger.store(
            **{
                'Loss/Loss_reward_critic': 0.0,
                'Loss/Loss_pi': 0.0,
                'Value/reward_critic1': 0.0,
            }
        )
        if self._cfgs.use_cost:
            self._logger.store(
                **{
                    'Loss/Loss_cost_critic': 0.0,
                    'Value/cost': 0.0,
                }
            )
