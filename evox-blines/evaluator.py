from collections.abc import Callable

import jax
import jax.numpy as jnp

from brax import envs

class BraxEvaluator:
    def __init__(
        self,
        policy: Callable,
        env_name: str,
        max_episode_length: int,
        num_episodes: int,
        backend: str = "generalized",
    ):
        """Contruct a brax-based problem

        Parameters
        ----------
        policy
            a callable: fn(weights, obs) -> action
        env_name
            The environment name.
        batch_size
            The number of brax environments to run in parallel.
            Usually this should match the population size at the algorithm side.
        max_episode_length
            The maximum number of timesteps of an episode.
        num_episodes
            Evaluating the number of episodes for each individual.
        backend
            Brax's backend, one of "generalized", "positional", "spring".
            Default to "generalized".
        """
        self.policy = policy
        self.env_name = env_name
        self.backend = backend
        self.env = envs.wrappers.training.VmapWrapper(
            envs.get_environment(env_name=env_name, backend=backend)
        )
        self.max_episode_length = max_episode_length
        self.num_episodes = num_episodes
        self.jit_reset = jax.jit(self.env.reset)
        self.jit_env_step = jax.jit(self.env.step)


    def evaluate(self, weights, key):

        def _cond_func(carry):
            counter, state, prev_done, _total_reward = carry
            return (counter < self.max_episode_length) & (~prev_done.all())

        def _body_func(carry):
            counter, brax_state, prev_done, total_reward = carry
            action = self.policy(weights, brax_state.obs)
            brax_state = self.jit_env_step(brax_state, action)
            total_reward += (1 - prev_done) * brax_state.reward
            done = jnp.logical_or(brax_state.done, prev_done)
            
            return counter + 1, brax_state, done, total_reward

        # create num_episodes parallel environments
        brax_state = self.jit_reset(
            jax.random.split(key, self.num_episodes)
        )

        # [pop_size, num_episodes]
        _, _, _, episode_retruns = jax.lax.while_loop(
            _cond_func,
            _body_func,
            (
                0,
                brax_state,
                jnp.zeros((self.num_episodes), dtype=jnp.bool),
                jnp.zeros((self.num_episodes)),
            ),
        )

        return episode_retruns