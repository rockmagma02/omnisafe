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
"""Safe Reinforcement Learning algorithms."""

import itertools
from types import MappingProxyType

from omnisafe.algorithms import model_based, off_policy, offline, on_policy

# Model-based Safe
from omnisafe.algorithms.model_based import CAP, MBPPOLag, SafeLOOP

# Off-Policy Safe
from omnisafe.algorithms.off_policy import DDPG, SAC, SDDPG, TD3, DDPGLag, SACLag, TD3Lag

# Offline Safe
from omnisafe.algorithms.offline import BCQ, VAEBC

# On-Policy Safe
from omnisafe.algorithms.on_policy import (
    CPO,
    CUP,
    FOCOPS,
    PCPO,
    PDO,
    PPO,
    TRPO,
    CPPOPid,
    NaturalPG,
    NPGLag,
    PolicyGradient,
    PPOEarlyTerminated,
    PPOLag,
    PPOLagEarlyTerminated,
    PPOLagSaute,
    PPOLagSimmerPid,
    PPOLagSimmerQ,
    PPOSaute,
    PPOSimmerPid,
    PPOSimmerQ,
    TRPOLag,
    TRPOPid,
)


ALGORITHMS = {
    'off-policy': tuple(off_policy.__all__),
    'offline': tuple(offline.__all__),
    'on-policy': tuple(on_policy.__all__),
    'model-based': tuple(model_based.__all__),
}

ALGORITHM2TYPE = {
    algo: algo_type for algo_type, algorithms in ALGORITHMS.items() for algo in algorithms
}

__all__ = ALGORITHMS['all'] = tuple(itertools.chain.from_iterable(ALGORITHMS.values()))

assert len(ALGORITHM2TYPE) == len(__all__), 'Duplicate algorithm names found.'

ALGORITHMS = MappingProxyType(ALGORITHMS)  # make this immutable
ALGORITHM2TYPE = MappingProxyType(ALGORITHM2TYPE)  # make this immutable

del itertools, MappingProxyType
