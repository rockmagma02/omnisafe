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
"""Implementation of GaussianStdNetActor."""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from omnisafe.models.base import Actor
from omnisafe.utils.model_utils import build_mlp_network


LOG_STD_MIN = -5
LOG_STD_MAX = 2


class GaussianStdNetActor(Actor):
    """Implementation of GaussianStdNetActor."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_max: torch.Tensor,
        act_min: torch.Tensor,
        hidden_sizes: list,
        activation,
        weight_initialization_mode,
        shared=None,
    ):
        """Initialize GaussianStdNetActor."""
        super().__init__(
            obs_dim, act_dim, hidden_sizes, activation, weight_initialization_mode, shared
        )
        self.act_min = act_min
        self.act_max = act_max

        if shared is not None:
            head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim * 2],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            self.net = nn.Sequential(shared, head)
        else:
            self.net = build_mlp_network(
                [obs_dim] + list(hidden_sizes) + [act_dim * 2],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )

    def _distribution(self, obs):
        out = self.net(obs)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (log_std + 1) * (LOG_STD_MAX - LOG_STD_MIN)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def predict(self, obs, deterministic=False, need_log_prob=False):
        dist = self._distribution(obs)
        if deterministic:
            out = dist.mean
        else:
            out = dist.rsample()

        action = torch.tanh(out)
        action = self.act_min + (action + 1) * 0.5 * (self.act_max - self.act_min)

        if need_log_prob:
            log_prob = dist.log_prob(out).sum(axis=-1)
            log_prob -= torch.log(1.00001 - torch.tanh(out).pow(2)).sum(axis=-1)
            return action.to(torch.float32), log_prob
        return action.to(torch.float32)

    def forward(self, obs, act=None):
        dist = self._distribution(obs)
        if act is not None:
            act = 2 * (act - self.act_min) / (self.act_max - self.act_min) - 1
            act = torch.atanh(torch.clamp(act, -0.999999, 0.999999))
            log_prob = dist.log_prob(act).sum(axis=-1)
            log_prob -= torch.log(1.00001 - torch.tanh(act).pow(2)).sum(axis=-1)
            return dist, log_prob
        return dist

    def change_device(self, device):
        """Change the device of the actor."""
        self.to(device)
        self.act_min = self.act_min.to(device)
        self.act_max = self.act_max.to(device)
