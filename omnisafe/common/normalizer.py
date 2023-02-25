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
"""Implementation of Vector Buffer."""

from typing import Tuple

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """Calculate normalized raw_data from running mean and std

    See  Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1979), "Updating Formulae and
    a Pairwise Algorithm for Computing Sample Variances." (PDF), Technical Report STAN-CS-79-773,
    Department of Computer Science, Stanford University.
    """

    def __init__(self, shape: Tuple[int, ...], device: torch.device, clip: float = 1e6) -> None:
        """Initialize the normalize."""
        super().__init__()
        if shape == ():
            self.register_buffer('_mean', torch.tensor(0.0, device=device))
            self.register_buffer('_sumsq', torch.tensor(0.0, device=device))
            self.register_buffer('_var', torch.tensor(0.0, device=device))
            self.register_buffer('_std', torch.tensor(0.0, device=device))
            self.register_buffer('_count', torch.tensor(0, device=device))
            self.register_buffer('_clip', clip * torch.tensor(1.0, device=device))
        else:
            self.register_buffer('_mean', torch.zeros(*shape, device=device))
            self.register_buffer('_sumsq', torch.zeros(*shape, device=device))
            self.register_buffer('_var', torch.zeros(*shape, device=device))
            self.register_buffer('_std', torch.zeros(*shape, device=device))
            self.register_buffer('_count', torch.tensor(0))
            self.register_buffer('_clip', clip * torch.ones(*shape, device=device))

        self._mean: torch.Tensor  # running mean
        self._sumsq: torch.Tensor  # running sum of squares
        self._var: torch.Tensor  # running variance
        self._std: torch.Tensor  # running standard deviation
        self._count: torch.Tensor  # number of samples
        self._clip: torch.Tensor  # clip value

        self._shape = shape
        self._device = device
        self._first = True

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the normalize."""
        return self._shape

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the normalize."""
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        """Return the std of the normalize."""
        return self._std

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize the data."""
        return self.normalize(data)

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize the _data."""
        data = data.to(self._mean.device)
        self._push(data)
        if self._count <= 1:
            return data
        output = (data - self._mean) / self._std
        return torch.clamp(output, -self._clip, self._clip)

    def _push(self, raw_data: torch.Tensor) -> None:
        if raw_data.shape == self._shape:
            raw_data = raw_data.unsqueeze(0)
        assert raw_data.shape[1:] == self._shape, 'data shape must be equal to (batch_size, *shape)'

        if self._first:
            self._mean = torch.mean(raw_data, dim=0)
            self._sumsq = torch.sum((raw_data - self._mean) ** 2, dim=0)
            self._count = torch.tensor(
                raw_data.shape[0], dtype=self._count.dtype, device=self._count.device
            )
            self._first = False
        else:
            count_raw = raw_data.shape[0]
            count = self._count + count_raw
            mean_raw = torch.mean(raw_data, dim=0)
            delta = mean_raw - self._mean
            self._mean += delta * count_raw / count
            sumq_raw = torch.sum((raw_data - mean_raw) ** 2, dim=0)
            self._sumsq += sumq_raw + delta**2 * self._count * count_raw / count
            self._count = count
        self._var = self._sumsq / (self._count - 1)
        self._std = torch.sqrt(self._var)
        self._std = torch.max(self._std, 1e-2 * torch.ones_like(self._std))
