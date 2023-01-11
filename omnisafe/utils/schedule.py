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
"""helper class to generate scheduling params"""


def _linear_interpolation(l, r, alpha):  # pylint: disable=invalid-name
    return l + alpha * (r - l)


# pylint: disable=too-few-public-methods
class PiecewiseSchedule:
    """Piecewise schedule for a value based on the step"""

    def __init__(self, endpoints, interpolation=_linear_interpolation, outside_value=None):
        """From OpenAI baselines"""
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, time):
        """Value at time t.

        Args:
            t (float): Time.

        Returns:
            float: Value at time t.
        """
        # pylint: disable=invalid-name
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= time < r_t:
                alpha = float(time - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class ConstantSchedule:
    """Constant schedule for a value"""

    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):  # pylint: disable=unused-argument, invalid-name
        """See Schedule.value"""
        return self._v
