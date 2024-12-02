from functools import partial
from typing import Optional
import gymnasium
from gymnasium import spaces
from gymnasium.vector.utils.spaces import batch_space

import jax
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu

from brax.envs.base import PipelineEnv

from brax import envs


class SyncVectorWrapper(gymnasium.vector.SyncVectorEnv):
    """Autoreset Vectorizes Brax env."""

    def __init__(
        self,
        env: PipelineEnv,
        num_envs: int,
        seed: int = 0,
        backend: Optional[str] = None,  # "cpu" or "gpu"
    ):
        self._env = env
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.dt,
        }

        self.num_envs = num_envs
        self.seed(seed)
        self.backend = backend
        self._state = None  # brax internal state

        obs = np.inf * np.ones(self._env.observation_size, dtype=np.float32)
        self.single_observation_space = spaces.Box(-obs, obs, dtype=np.float32)
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )

        action = jax.tree.map(np.array, self._env.sys.actuator.ctrl_range)
        self.single_action_space = spaces.Box(
            action[:, 0], action[:, 1], dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.episodic_return = jnp.zeros(self.num_envs)  # 初始化奖励累积
        self.episodic_length = jnp.zeros(self.num_envs)  # 初始化步数累积

        def reset(key):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset, backend=self.backend)

        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step, backend=self.backend)

    def step(self, action):
        if self._state is None:
            raise ValueError("Cannot call step() before calling reset()")

        self._state, obs, reward, done, info = self._step(self._state, action)

        self.episodic_return += reward  # 累积奖励
        self.episodic_length += 1  # 累积步数

        if jnp.any(done):
            final_info = [None] * self.num_envs
            final_observation = [None] * self.num_envs

            reset_state, reset_obs, self._key = self._reset(self._key)

            # gymnasium style info
            for i, d in enumerate(done):
                if d:
                    final_info[i] = {
                        "episode": {
                            "r": self.episodic_return[i],
                            "l": self.episodic_length[i],
                        }
                    }
                    final_observation[i] = obs[i]
                    self.episodic_return = self.episodic_return.at[i].set(0)
                    self.episodic_length = self.episodic_length.at[i].set(0)

            info["final_info"] = final_info
            info["final_observation"] = final_observation

            # Autoreset
            self._state = jtu.tree_map(
                partial(_where_done, done), reset_state, self._state
            )
            obs = _where_done(done, reset_obs, obs)

        truncation = info["truncation"].astype(jnp.bool)
        termination = jnp.logical_and(done, ~truncation)

        return obs, reward, termination, truncation, info

    def reset(self, *, seed: Optional[int] = None):
        if seed is not None:
            self.seed(seed)
        self._state, obs, self._key = self._reset(self._key)
        info = {}
        return obs, info

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)


def _where_done(done, x, y):
    if done.ndim > 0:
        done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
    return jnp.where(done, x, y)


def make_env(env_id, num_envs, seed):
    # Note: autoreset is handled in SyncVectorWrapper
    env = envs.create(
        env_id, batch_size=num_envs, episode_length=1000, auto_reset=False
    )
    env = SyncVectorWrapper(env, num_envs=num_envs, seed=seed)
    return env
