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
"""Implementation of BCQ Actor."""

import torch
from torch import nn

from omnisafe.utils.model_utils import build_mlp_network


class BCQActor(nn.Module):
    """Implementation of BCQActor."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_dim,
        act_dim,
        act_min: torch.Tensor,
        act_max: torch.Tensor,
        phi: float,
        hidden_sizes,
        activation,
        weight_initialization_mode,
    ):
        """Initialize BCQActor."""
        super().__init__()
        self.act_min = act_min
        self.act_max = act_max

        self.net = build_mlp_network(
            sizes=[obs_dim + act_dim] + hidden_sizes + [act_dim],
            activation=activation,
            output_activation='tanh',
            weight_initialization_mode=weight_initialization_mode,
        )

        self.phi = phi

    def forward(self, obs, act):
        """Forward pass of the model.

        Args:
            obs (Tensor): Observation.
            act (Tensor): Action.

        Returns:
            Tensor: Action.
        """
        out = self.net(torch.cat([obs, act], dim=-1))
        out = self.act_min + (self.act_max - self.act_min) * (out + 1) / 2
        return (act + self.phi * out).clamp(self.act_min, self.act_max)

    def change_deivce(self, device):
        """Change device of the model.

        Args:
            device (str): Device to change to.
        """
        self.net.to(device)
        self.act_min = self.act_min.to(device)
        self.act_max = self.act_max.to(device)
