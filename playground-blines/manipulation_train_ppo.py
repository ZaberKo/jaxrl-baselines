import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

import argparse
from pprint import pprint
import numpy as np
from pathlib import Path
import mediapy as media
import matplotlib.pyplot as plt

import wandb
from datetime import datetime
import functools
import os
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

import jax
from jax import numpy as jp
import chex
import mediapy as media
import numpy as np

from mujoco_playground import wrapper
from mujoco_playground import registry

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO agent on playground.")
    parser.add_argument(
        "--env_name",
        type=str,
        default="PandaPickCube",
        help="Name of the environment to train on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training.",
    )
    return parser.parse_args()


def train_ppo(args):
    env_name = args.env_name
    seed = args.seed

    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)

    print("env_cfg:")
    print(env_cfg)

    from mujoco_playground.config import manipulation_params

    ppo_params = manipulation_params.brax_ppo_config(env_name)
    ppo_params.num_evals = 100
    ppo_params.num_timesteps = 2_000_000
    print("ppo_params:")
    print(ppo_params)

    config = dict(
        env_cfg=env_cfg,
        ppo_params=ppo_params,
    )
    tags = ["ppo", "playground"]

    output_dir = Path(
        f"./outputs/ppo-{env_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    wandb.init(
        project="JAXRL-baselines",
        name=f"PPO-playground-{env_name}",
        config=config,
        tags=tags,
        dir=output_dir,
    )

    times = [datetime.now()]

    def metrics_todict(metrics):
        return jax.tree_util.tree_map(
            lambda x: x.tolist() if isinstance(x, chex.Array) else x, metrics
        )

    def wandb_progess_fn(env_steps, metrics):
        times.append(datetime.now())
        metrics = metrics_todict(metrics)
        print(f"env_steps: {env_steps}")
        pprint(metrics)
        wandb.log(metrics, env_steps)

    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )

    train_fn = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=wandb_progess_fn,
        seed=seed,
    )

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    wandb.finish()

    video_path = output_dir / f"{env_name}.mp4"
    print(f"Rendering to {video_path}")
    visualize(env, make_inference_fn, params, video_path)


def visualize(env, make_inference_fn, params, path):
    env_cfg = env._config
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

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
    media.write_video(path, frames, fps=1.0 / env.dt / render_every)
    print(f"reward: {np.sum(rewards)}")


if __name__ == "__main__":
    args = parse_args()
    train_ppo(args)
