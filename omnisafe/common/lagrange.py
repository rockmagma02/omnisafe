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
"""Implementation of Lagrange."""

import abc

import torch


class Lagrange(abc.ABC):
    """Abstract base class for Lagrangian-base Algorithms."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str,
        lagrangian_upper_bound=None,
        beta: float = 1.0,
        device: torch.device = torch.device('cpu'),
    ):
        """init"""
        self.cost_limit = cost_limit * beta
        self.lambda_lr = lambda_lr
        self.lagrangian_upper_bound = lagrangian_upper_bound

        init_value = max(lagrangian_multiplier_init, 1e-5)
        self.lagrangian_multiplier = torch.nn.Parameter(
            torch.as_tensor(init_value, device=device), requires_grad=True
        )
        self.lambda_range_projection = torch.nn.ReLU()
        # fetch optimizer from PyTorch optimizer package
        assert hasattr(
            torch.optim, lambda_optimizer
        ), f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(torch.optim, lambda_optimizer)
        self.lambda_optimizer = torch_opt(
            [
                self.lagrangian_multiplier,
            ],
            lr=lambda_lr,
        )

    def compute_lambda_loss(self, mean_ep_cost):
        """Penalty loss for Lagrange multiplier."""
        return -self.lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

    def update_lagrange_multiplier(self, Jc):
        """Update Lagrange multiplier (lambda)
        Note: Jc obtained from: self.logger.get_stats('EpCosts')[0]
        are already averaged across MPI processes.
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(
            0, self.lagrangian_upper_bound
        )  # enforce: lambda in [0, inf]
