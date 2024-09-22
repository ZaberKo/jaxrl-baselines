#%%
import brax.envs

env = brax.envs.create("ant")
# %%
from brax.envs.wrappers.gym import VectorGymWrapper

gym_env = VectorGymWrapper(env)
# %%
# import os
# if len(os.environ.get("LD_LIBRARY_PATH", "")) ==0:
#     os.environ["LD_LIBRARY_PATH"]="/home/zaber/.mujoco/mujoco210/bin:/usr/lib/nvidia"

import numpy as np
import gymnasium

from gymnasium.vector import SyncVectorEnv
from pprint import pprint

sync_env = SyncVectorEnv([lambda: gymnasium.make("Ant-v4")]*7)

obs, infos = sync_env.reset(seed=0)

pprint(infos)
#%%

history = []

for i in range(1000):
    action = sync_env.action_space.sample()
    data = sync_env.step(action)
    history.append(data)
    obs, reward, terminated, trunc, info = data

    if np.any(terminated):
        break

# %%
