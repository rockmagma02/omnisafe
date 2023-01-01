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
"""Implementation of VAE."""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from omnisafe.utils.model_utils import build_mlp_network


class VAE(nn.Module):
    """Class for VAE."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        latent_dim,
        activation,
        weight_initialization_mode,
    ):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim

        self._encoder = build_mlp_network(
            sizes=[obs_dim + act_dim] + hidden_sizes + [latent_dim * 2],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self._decoder = build_mlp_network(
            sizes=[obs_dim + latent_dim] + hidden_sizes + [act_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def encode(self, obs, act):
        """Encode observation to latent space."""
        latent = self._encoder(torch.cat([obs, act], dim=-1))
        mean, log_std = torch.chunk(latent, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return Normal(mean, log_std.exp())

    def decode(self, obs, latent=None):
        """Decode latent space to action."""
        if latent is None:
            latent = Normal(0, 1).sample([obs.shape[0], self.latent_dim]).to(obs.device)

        return self._decoder(torch.cat([obs, latent], dim=-1))

    def loss(self, obs, act):
        """Compute loss for VAE."""
        dist = self.encode(obs, act)
        latent = dist.rsample()
        pred_act = self.decode(obs, latent)
        recon_loss = nn.functional.mse_loss(pred_act, act)
        kl_loss = torch.distributions.kl.kl_divergence(dist, Normal(0, 1)).mean()
        return recon_loss, kl_loss

    def forward(self, obs, act):
        """Forward function for VAE."""
        dist = self.encode(obs, act)
        latent = dist.rsample()
        pred_act = self.decode(obs, latent)
        return pred_act, dist, latent
