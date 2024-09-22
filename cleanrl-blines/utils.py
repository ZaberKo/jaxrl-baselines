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

# class Evaluator_for_gymnasium:
#     def __init__(self, env, num_envs, seed):
#         # 初始化并行环境
#         self.env = env
#         self.num_envs = num_envs
#         self.key = jax.random.PRNGKey(seed)

#     def evaluate(self, actor, actor_state):
#         key, env_key = jax.random.split(self.key)
#         env_seed = int(jax.random.randint(env_key, (), 0, 2**15))
#         obs, _ = self.env.reset(seed=env_seed)
#         rewards = []
#         lengths = []
#         self.key = key
#         actor_apply = jax.jit(actor.apply)  # JIT 编译

#         while len(rewards) < self.num_envs:
#             # 计算动作
#             actions = actor_apply(actor_state.params, obs)
#             actions = jax.device_get(actions)
#             # 执行动作
#             next_obs, _, _, _, infos = self.env.step(actions)

#             if "final_info" in infos:
#                 for info in infos["final_info"]:
#                     if info and "episode" in info:
#                         rewards.append(info["episode"]["r"])
#                         lengths.append(info["episode"]["l"])

#             obs = next_obs  # 更新当前的状态

#         rewards = jnp.array(rewards)
#         lengths = jnp.array(lengths)

#         return jnp.mean(rewards), jnp.mean(lengths)


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
