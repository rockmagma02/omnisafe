import multiprocessing
import sys
import os
from copy import deepcopy
from typing import Dict, List

import torch

import omnisafe

USE_CUDA = False
# CPU config
PER_THREADS = 6
MAX_SUBPROCESSES = 11
# GPU config
PROSESS_PER_GPU = 3
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

USE_REDIRECTION = True

grid_cfgs = {
    'env_id': ['SafetyPointGoal1-v0'],
    'algo': ['CPO', 'PPO', 'PPOLag']
}

class Mission:
    env_id: str
    algo: str
    custom_cfgs: dict

def cfgs2mission(grid_cfgs: Dict[str, list]) -> List[Mission]:
    num_items = len(grid_cfgs.items())
    missions = [grid_cfgs]
    for idx in range(num_items):
        new_missions = []
        for mission in missions:
            key, values = list(mission.items())[idx]
            for value in values:
                new_mission = deepcopy(mission)
                new_mission[key] = value
                new_missions.append(new_mission)
        missions = new_missions

    def dict2mission(mission: dict) -> Mission:
        m = Mission()
        m.env_id = mission['env_id']
        m.algo = mission['algo']
        m.custom_cfgs = {k: v for k, v in mission.items() if k not in ['env_id', 'algo']}
        return m

    return [dict2mission(mission) for mission in missions]

def train(mission: Mission, device: int = None, mission_id: int = None):
    if USE_CUDA:
        device = f'cuda:{device}'
        torch.cuda.set_device(device)
        torch.set_num_threads(1)
        mission.custom_cfgs['device'] = device
    else:
        torch.set_num_threads(PER_THREADS)

    if USE_REDIRECTION:
        if not os.path.exists('./runs/log'):
            os.mkdir('./runs/log')
        sys.stdout = open(f'./runs/log/{mission.algo}_{mission.env_id}_{mission_id}.log', 'w')

    agent = omnisafe.Agent(mission.algo, mission.env_id, custom_cfgs=mission.custom_cfgs)
    agent.learn()


if __name__ == '__main__':
    missions = cfgs2mission(grid_cfgs)
    if USE_CUDA:
        tasks = {k: 0 for k in GPUS}
        pool = multiprocessing.Pool(processes=len(GPUS) * PROSESS_PER_GPU)
        for idx, mission in enumerate(missions):
            device = min(tasks, key=tasks.get)
            tasks[device] += 1
            pool.apply_async(train, args=(mission, device, idx))
        pool.close()
        pool.join()
    else:
        pool = multiprocessing.Pool(processes=MAX_SUBPROCESSES)
        for idx, mission in enumerate(missions):
            pool.apply_async(train, args=(mission, None, idx))
        pool.close()
        pool.join()
