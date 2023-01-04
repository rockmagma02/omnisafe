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
"""Implementation of model DICE algo family."""

import torch
from torch import nn

from omnisafe.utils.model_utils import Activation, InitFunction, build_mlp_network


class ObsDecoder(nn.Module):
    """Abstract base class for observation decoder."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: int,
        out_dim: int,
        hidden_sizes: list,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'xavier_uniform',
    ):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.out_dim = out_dim
        self.weight_initialization_mode = weight_initialization_mode
        self.activation = activation
        self.hidden_sizes = hidden_sizes
        self.net = build_mlp_network(
            [obs_dim] + list(hidden_sizes) + [out_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs (torch.Tensor): Observation.

        Returns:
            torch.Tensor: Decoded observation.
        """
        if self.out_dim == 1:
            return self.net(obs).squeeze(-1)
        return self.net(obs)
