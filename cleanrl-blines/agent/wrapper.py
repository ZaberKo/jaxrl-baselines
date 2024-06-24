# Copyright 2024 The Brax Authors.
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

# pylint:disable=g-multiple-import
"""A brax environment for training and inference."""

import abc
from typing import Any, Dict, List, Optional, Sequence, Union

from brax import base
from brax.generalized import pipeline as g_pipeline
from brax.io import image
from brax.positional import pipeline as p_pipeline
from brax.spring import pipeline as s_pipeline
from flax import struct
import jax
import numpy as np
import jax.numpy as jnp
from brax.envs.wrappers.gym import GymWrapper, VectorGymWrapper
from brax.envs.base import Wrapper
from brax import envs
import gymnasium as gym
import flashbax
import warnings


class GymnasiumWrapper(VectorGymWrapper):
    """Vectorizes Brax env."""
    def setup(self):
        self.single_action_space = self.action_space
        self.single_observation_space = self.observation_space
        self._step = jax.jit(self._step)

    def step(self, action):
        action = jnp.squeeze(action, axis=0)
        self._state, obs, reward, done, info = self._step(self._state, action)
        if done or info["truncation"]:
            obs, _ = self.reset()
        return obs, reward, done, info["truncation"], info

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self._state, obs, self._key = self._reset(self._key)
        return obs, self._state.info
    
    def close(self):
        return None


def make_env(env_id, seed, idx, capture_video, run_name, num_envs):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="brax.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="flashbax.*")
    env = envs.create(env_id, batch_size=num_envs, episode_length=1000)
    env = GymnasiumWrapper(env, seed=seed)
    env.setup()
    return env


class ReplayBuffer():

    def __init__(self, buffer_size, env, batch_size, key):
        self.rb, self.rb_state = self._replayer_buffer(
            buffer_size, batch_size, env
        )
        self.key = key
        # self.enable_jit()

    def _replayer_buffer(self, buffer_size, batch_size, env):
        rb = flashbax.make_item_buffer(
            max_length=buffer_size,
            min_length=buffer_size,
            sample_batch_size=batch_size,
            add_batches=True,
        )
        dummy_action = np.squeeze(env.action_space.sample(),axis=0)
        dummy_obs = np.squeeze(env.observation_space.sample(),axis=0)
        dummy_reward = jnp.zeros(())
        dummy_done = jnp.zeros(())
        dummy_next_obs = dummy_obs

        dummy_sample_batch = self._get_rb_item(
            dummy_obs,
            dummy_action,
            dummy_reward,
            dummy_next_obs,
            dummy_done,
            dummy_done,
        )
        rb_state = rb.init(dummy_sample_batch)
        return rb, rb_state

    def enable_jit(self):
        self.add = jax.jit(self.add)
        # self.sample = jax.jit(self.sample)

    def add(self, obs, real_next_obs, actions, rewards, terminations, infos):
        actions = jnp.squeeze(actions, axis=0)

        # print("Expected structure:", jax.tree_util.tree_structure(self.rb_state.experience['next_obs']))  # Adjust according to your state structure
        item = self._get_rb_item(
            obs, actions, rewards, real_next_obs, terminations, infos["truncation"]
        )
        self.rb_state = self.rb.add(self.rb_state, item)

    def sample(self, batch_size=None):
        self.key, key = jax.random.split(self.key)
        return self.rb.sample(self.rb_state, key).experience

    def _get_rb_item(self, obs, action, reward, next_obs, done, truncation):
        item = dict(
            observations=obs,
            actions=action,
            rewards=reward,
            next_observations=next_obs,
            dones=done,
            truncations=truncation,
        )
        return item
