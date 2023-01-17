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
"""Implementation of VCritic."""

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.utils.model_utils import Activation, InitFunction, build_mlp_network


class VCritic(Critic):
    """Implementation of VCritic."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared: nn.Module = None,
        num_critics: int = 1,
    ) -> None:
        """Initialize."""
        self.num_critics = num_critics
        Critic.__init__(
            self,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            shared=shared,
        )
        self.critic_list = []
        for idx in range(num_critics):
            net = build_mlp_network(
                [obs_dim] + hidden_sizes + [1],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            critic = nn.Sequential(net)
            self.critic_list.append(critic)
            self.add_module(f'critic_{idx}', critic)
        self.net = self.critic_list[0]

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward."""
        res = []
        for critic in self.critic_list:
            res.append(torch.squeeze(critic(obs), dim=-1))
        if self.num_critics == 1:
            return res[0]
        return res
