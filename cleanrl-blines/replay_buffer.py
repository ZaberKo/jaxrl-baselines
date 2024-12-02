from typing import NamedTuple

import jax
import jax.numpy as jnp

import flashbax


class ReplayBufferSamples(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    next_observations: jax.Array
    dones: jax.Array
    rewards: jax.Array


class ReplayBuffer:
    """
    A jax.Array based Replay Buffer: supported on GPU/TPU
    """

    def __init__(
        self,
        buffer_size,
        batch_size,
        observation_space,
        action_space,
        key,
        enable_jit=True,
    ):
        self.rb = flashbax.make_item_buffer(
            max_length=buffer_size,
            min_length=batch_size,
            sample_batch_size=batch_size,
            add_batches=True,
        )
        dummy_action = jnp.asarray(action_space.sample())
        dummy_obs = jnp.asarray(observation_space.sample())
        dummy_reward = jnp.zeros(())
        dummy_done = jnp.zeros((), dtype=jnp.bool)
        dummy_next_obs = dummy_obs

        dummy_sample_batch = self._create_sample_batch(
            dummy_obs,
            dummy_next_obs,
            dummy_action,
            dummy_reward,
            dummy_done,
        )
        self.rb_state = self.rb.init(dummy_sample_batch)
        self.key = key

        self._add = self.rb.add
        self._sample = self.rb.sample

        if enable_jit:
            self._add = jax.jit(self._add)
            self._sample = jax.jit(self._sample)

    def add(self, obs, next_obs, actions, rewards, dones):
        item = self._create_sample_batch(obs, next_obs, actions, rewards, dones)
        self.rb_state = self._add(self.rb_state, item)

    def sample(self):
        self.key, key = jax.random.split(self.key)
        sample_batch = self._sample(self.rb_state, key).experience
        return sample_batch

    def _create_sample_batch(self, obs, next_obs, actions, rewards, dones):
        return ReplayBufferSamples(
            observations=obs,
            next_observations=next_obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )
