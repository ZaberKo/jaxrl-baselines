import jax
import numpy as np
import jax.numpy as jnp
from brax.envs.wrappers.gym import GymWrapper, VectorGymWrapper
from brax import envs
import flashbax
import warnings


class GymnasiumWrapper(VectorGymWrapper):
    """Vectorizes Brax env."""

    def setup(self):
        self.single_action_space = self.action_space
        self.single_observation_space = self.observation_space
        self._step = jax.jit(self._step)
        self._reset = jax.jit(self._reset)
        self.episodic_return = jnp.zeros(self.num_envs)  # 初始化奖励累积
        self.episodic_length = jnp.zeros(self.num_envs)  # 初始化步数累积

    def step(self, action):
        action = jnp.squeeze(action, axis=0)
        self._state, _obs, reward, done, _info = self._step(self._state, action)

        self.episodic_return += reward  # 累积奖励
        self.episodic_length += 1  # 累积步数

        # 只重置那些done或truncation的环境
        reset_mask = jnp.logical_or(done, _info["truncation"])
        if jnp.any(reset_mask):
            final_info = []
            for i in range(self._state.obs.shape[0]):
                if reset_mask[i]:
                    final_info.append({
                        "episode": {
                            "r": self.episodic_return[i],
                            "l": self.episodic_length[i],
                        }
                    })
                    self.episodic_return = self.episodic_return.at[i].set(0)
                    self.episodic_length = self.episodic_length.at[i].set(0)
            obs, _info = self.reset(reset_mask)
            info = _info.copy()
            info["final_info"] = final_info
        else:
            obs = _obs
            info = _info.copy()
        info["final_observation"] = _obs

        return obs, reward, done, info["truncation"], info

    def reset(self, reset_mask=None, seed=None):
        if seed is not None:
            self.seed(seed)
        
        if reset_mask is None:
            # 全部重置
            self._state, obs, self._key = self._reset(self._key)
        else:
            # 只重置需要重置的部分环境
            reset_state, reset_obs, self._key = self._reset(self._key)
            self._state = jax.tree_map(
                lambda x, y: jnp.where(reset_mask, x, y),
                reset_state,
                self._state
            )
            obs = jax.tree_map(
                lambda x, y: jnp.where(reset_mask, x, y),
                reset_obs,
                self._state.obs
            )

        return obs, self._state.info

    def close(self):
        return None


def make_env(env_id, seed, num_envs):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="brax.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="flashbax.*")
    env = envs.create(env_id, batch_size=num_envs, episode_length=1000)
    env = GymnasiumWrapper(env, seed=seed)
    env.setup()
    return env


class ReplayBuffer:

    def __init__(self, buffer_size, env, batch_size, key, save_info=False):
        self.rb, self.rb_state = self._replayer_buffer(buffer_size, batch_size, env)
        self.key = key
        # self.enable_jit()

    def _replayer_buffer(self, buffer_size, batch_size, env):
        rb = flashbax.make_item_buffer(
            max_length=buffer_size,
            min_length=buffer_size,
            sample_batch_size=batch_size,
            add_batches=True,
        )
        dummy_action = jnp.squeeze(env.action_space.sample(), axis=0)
        dummy_obs = jnp.squeeze(env.observation_space.sample(), axis=0)
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

    def add(self, obs, real_next_obs, actions, rewards, terminations, truncation):
        actions = jnp.squeeze(actions, axis=0)

        # print("Expected structure:", jax.tree_util.tree_structure(self.rb_state.experience['next_obs']))  # Adjust according to your state structure
        item = self._get_rb_item(
            obs, actions, rewards, real_next_obs, terminations, truncation
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
