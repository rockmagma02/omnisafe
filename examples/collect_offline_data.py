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
"""Example of collecting offline data."""

from omnisafe.utils.offline_collector import Collector


if __name__ == '__main__':
    col = Collector(
        env_id='SafetyCarRun0-v0',
        size=2_000_000,
        random_proportion=1.0,
        expert_proportion=0.0,
        expert_path='./runs/SafetyCarRun0-v0/CPO/seed-000-2023-01-01_03-59-02',
        expert_model_name='model 499.pt',
        unsafe_proportion=0.0,
        unsafe_path='./runs/SafetyCarRun0-v0/PPO/seed-000-2023-01-01_02-05-24',
        unsafe_model_name='model 299.pt',
        noise_std=0.3,
    )
    col.collect()
    save_name = 'SafetyCarRun0-v0-random-std03'
    col.save('./runs/data/', save_name)
