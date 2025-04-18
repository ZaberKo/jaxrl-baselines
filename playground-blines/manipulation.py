#%%
import os
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
os.environ["MUJCOCO_GL"] = "egl"


import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# %%
# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp

from mujoco_playground import wrapper
from mujoco_playground import registry


# %%
env_name = 'PandaPickCube'
# env_name = 'PandaOpenCabinet'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)

env_cfg

# %% [markdown]
# ## Train Policy
# 
# Let's train the pick cube policy and visualize rollouts. The policy takes roughly 3 minutes to train on an RTX 4090.

# %%
from mujoco_playground.config import manipulation_params
ppo_params = manipulation_params.brax_ppo_config(env_name)
ppo_params.num_evals = 40
ppo_params

# %% [markdown]
# ### PPO

# %%
from pprint import pprint
from collections import defaultdict 
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


loss_data = defaultdict(list)

def progress(num_steps, metrics):
  clear_output(wait=True)

  pprint(metrics)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(5*4, 3))
  
  axs[0].set_xlim([0, ppo_params["num_timesteps"] * 1.25])
  axs[0].set_xlabel("# environment steps")
  axs[0].set_ylabel("reward per episode")
  axs[0].set_title(f"y={y_data[-1]:.3f}")
  axs[0].errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

  if "training/policy_loss" in metrics:
    
    loss_data["policy_loss"].append(metrics["training/policy_loss"])
    loss_data["v_loss"].append(metrics["training/v_loss"])
    loss_data["entropy_loss"].append(metrics["training/entropy_loss"])

    for i, (name, data) in enumerate(loss_data.items(), 1):
      axs[i].set_xlim([0, ppo_params["num_timesteps"] * 1.25])
      axs[i].set_xlabel("# environment steps")
      axs[i].set_ylabel(name)
      axs[i].plot(x_data[1:], data, color="blue")
      axs[i].set_title(f"y={data[-1]:.3e}")

  fig.tight_layout()
  display(fig)

ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    progress_fn=progress,
    # seed=1
    seed=42
)

# %%
make_inference_fn, params, metrics = train_fn(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

# %% [markdown]
# ## Visualize Rollouts

# %%
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

# %%
rng = jax.random.PRNGKey(42)
rollout = []
n_episodes = 1

for _ in range(n_episodes):
  state = jit_reset(rng)
  rollout.append(state)
  for i in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)

render_every = 1
frames = env.render(rollout[::render_every])
rewards = [s.reward for s in rollout]
media.show_video(frames, fps=1.0 / env.dt / render_every)

# %% [markdown]
# While the above policy is very simple, the work was extended using the Madrona batch renderer, and policies were transferred on a real robot. We encourage folks to check out the Madrona-MJX tutorial notebooks ([part 1](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1.ipynb) and [part 2](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_2.ipynb))!

# %% [markdown]
# # Dexterous Manipulation
# 
# Let's now train a policy that was transferred onto a real Leap Hand robot with the `LeapCubeReorient` environment! The environment contains a cube placed in the center of the hand, and the goal is to re-orient the cube in SO(3).

# %%
env_name = 'LeapCubeReorient'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)

# %%
env_cfg

# %% [markdown]
# ## Train Policy
# 
# Let's train an initial policy and visualize the rollouts. Notice that the PPO parameters contain `policy_obs_key` and `value_obs_key` fields, which allow us to train brax PPO with [asymmetric](https://arxiv.org/abs/1710.06542) observations for the actor and the critic. While the actor recieves proprioceptive state similar in nature to the real-world camera tracking sensors, the critic network recieves privileged state only available in the simulator. This enables more sample efficient learning, and we are able to train an initial policy in 33 minutes on a single RTX 4090.
# 
# Depending on the GPU device and topology, training can be brought down to 10-20 minutes as shown in the MuJoCo Playground technical report.

# %%
from mujoco_playground.config import manipulation_params
ppo_params = manipulation_params.brax_ppo_config(env_name)
ppo_params

# %% [markdown]
# ### PPO

# %%
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):
  clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

  display(plt.gcf())

ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    progress_fn=progress,
    seed=1
)

# %%
make_inference_fn, params, metrics = train_fn(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

# %% [markdown]
# ## Visualize Rollouts

# %%
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

# %%
rng = jax.random.PRNGKey(42)
rollout = []
n_episodes = 1

for _ in range(n_episodes):
  state = jit_reset(rng)
  rollout.append(state)
  for i in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)

render_every = 1
frames = env.render(rollout[::render_every])
rewards = [s.reward for s in rollout]
media.show_video(frames, fps=1.0 / env.dt / render_every)

# %% [markdown]
# The above policy solves the task, but may look a little bit jittery. To get robust sim-to-real transfer,  we retrained from previous checkpoints using a curriculum on the maximum torque to facilitate exploration early on in the curriculum, and to produce smoother actions for the final policy. More details can be found in the MuJoCo Playground technical report!

# %% [markdown]
# 🙌 Thanks for stopping by The Playground!


