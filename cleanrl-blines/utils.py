from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import re
from pathlib import Path
from functools import partial

import brax
from brax import envs

import jax.numpy as jnp
import jax


def set_omegaconf_resolvers():
    OmegaConf.register_new_resolver(
        "sanitize_dirname", lambda path: re.sub(r"/", "_", path)
    )


def get_output_dir(default_path: str = "./debug"):
    if HydraConfig.initialized():
        output_dir = Path(HydraConfig.get().runtime.output_dir).absolute()
    else:
        output_dir = Path(default_path).absolute()

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    return output_dir


def yaml_print(data):
    print(OmegaConf.to_yaml(data))


class BraxEvaluator:
    def __init__(
        self, env_id, action_fn, num_episodes=10, max_episode_length=1000, **env_kwargs
    ):

        self.env = envs.wrappers.training.VmapWrapper(
            envs.get_environment(env_name=env_id, **env_kwargs)
        )
        self.num_envs = num_episodes
        self.max_episode_length = max_episode_length

        self.action_fn = action_fn

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, actor_state, key):
        env_state = self.env.reset(jax.random.split(key, self.num_envs))

        def _cond_fn(carry):
            counter, env_state, prev_done, episode_returns, episode_lengths = carry
            return (counter < self.max_episode_length) & (~prev_done.all())

        def _body_fn(carry):
            counter, env_state, prev_done, episode_returns, episode_lengths = carry
            action = self.action_fn(actor_state.params, env_state.obs)
            env_state = self.env.step(env_state, action)

            # Note: use jnp.where to avoid inf in env_state
            rewards = jnp.where(
                prev_done, jnp.zeros_like(env_state.reward), env_state.reward
            )
            episode_returns = episode_returns + rewards

            done = jnp.where(prev_done, prev_done, env_state.done.astype(jnp.bool))
            episode_lengths = episode_lengths + (1 - done)

            return (counter + 1, env_state, done, episode_returns, episode_lengths)

        init_vals = (
            0,
            env_state,
            jnp.zeros(self.num_envs, dtype=jnp.bool),
            jnp.zeros(self.num_envs),
            jnp.zeros(self.num_envs),
        )

        _, _, _, episode_returns, episode_lengths = jax.lax.while_loop(
            _cond_fn, _body_fn, init_vals
        )

        return episode_returns, episode_lengths
