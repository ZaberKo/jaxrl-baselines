import os
import random
import time
from dataclasses import dataclass

import flax
import flax.linen as nn
from brax import envs as brax_envs
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import optax
# import tyro
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter
from omegaconf import DictConfig
from brax.envs.wrappers import gym as gym_wrapper
import warnings
import flashbax

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def get_rb_item(obs, action, reward, next_obs, done, truncation):
    item = dict(
    obs=obs,
    actions = action,
    rewards = reward,
    next_obs = next_obs,
    dones = done,
    truncation = truncation)
    return item

def get_rb_item_from_state(env_state, obs, action):
    item = dict(
        obs=obs,
        actions=action,
        rewards=env_state.reward,
        next_obs=env_state.obs,
        dones=env_state.done,
        truncation=env_state.info["truncation"],
    )
    return item


def replayer_buffer(args, env):
    rb = flashbax.make_item_buffer(
        max_length=args.buffer_size,
        min_length=args.buffer_size,
        sample_batch_size=args.batch_size,
        add_batches=True,
    )
    dummy_action = jnp.zeros(env.action_size)
    dummy_obs = jnp.zeros(env.observation_size)

    dummy_reward = jnp.zeros(())
    dummy_done = jnp.zeros(())
    dummy_nest_obs = dummy_obs

    dummy_sample_batch = get_rb_item(
        dummy_obs, dummy_action, dummy_reward, dummy_nest_obs, dummy_done, dummy_done
    )
    rb_state = rb.init(dummy_sample_batch)
    return rb, rb_state

def main(args: DictConfig) -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="brax.*") 
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="flashbax.*")
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key, env_key = jax.random.split(key, 4)

    # env setup
    envs = brax_envs.create(env_name=args.env_name, batch_size=args.num_envs, )
    # envs.observation_space.dtype = np.float32
    rb, rb_state = replayer_buffer(args, envs)

    # TRY NOT TO MODIFY: start the game
    env_state = envs.reset(env_key)
    action_space = envs.sys.actuator.ctrl_range
    obs = env_state.obs
    action_low = action_space[0, 0]
    action_high = action_space[0, 1]
    actor = Actor(
        action_dim=envs.action_size,
        action_scale=(action_space[0, 1] - action_space[0, 0]) / 2,
        action_bias=(action_space[0, 1] + action_space[0, 0]) / 2,
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    qf = QNetwork()
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs, jnp.ones((1, envs.action_size))),
        target_params=qf.init(qf1_key, obs, jnp.ones((1, envs.action_size))),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
    ):
        next_state_actions = (
            actor.apply(actor_state.target_params, next_observations)
        ).clip(
            -1, 1
        )  # TODO: proper clip
        qf1_next_target = qf.apply(
            qf1_state.target_params, next_observations, next_state_actions
        ).reshape(-1)
        next_q_value = (
            rewards + (1 - terminations) * args.gamma * (qf1_next_target)
        ).reshape(-1)

        def mse_loss(params):
            qf_a_values = qf.apply(params, observations, actions).squeeze()
            return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(
            mse_loss, has_aux=True
        )(qf1_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads1)

        return qf1_state, qf1_loss_value, qf1_a_values

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
    ):
        def actor_loss(params):
            return -qf.apply(
                qf1_state.params, observations, actor.apply(params, observations)
            ).mean()

        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(
                actor_state.params, actor_state.target_params, args.tau
            )
        )

        qf1_state = qf1_state.replace(
            target_params=optax.incremental_update(
                qf1_state.params, qf1_state.target_params, args.tau
            )
        )
        return actor_state, qf1_state, actor_loss_value

    start_time = time.time()
    envs.step = jit(envs.step)
    # rb.add = jit(rb.add)
    for global_step in range(args.total_timesteps):
        key, loop_key, sample_key = jax.random.split(key, 3)
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            # actions = np.array(
            #     [envs.action_space.sample() for _ in range(args.num_envs)]
            # )
            actions = jax.random.uniform(
                loop_key,
                shape=(args.num_envs, envs.action_size),
                minval=action_low,
                maxval=action_high,
            )
        else:
            actions = actor.apply(actor_state.params, obs)
            actions = np.array(
                [
                    (
                        jax.device_get(actions)[0]
                        + np.random.normal(
                            0, actor.action_scale * args.exploration_noise
                        )
                    ).clip(action_low, action_high)
                ]
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        env_state = envs.step(env_state, actions)
        next_obs, rewards, terminations, truncations, infos = (
            env_state.obs,
            env_state.reward,
            env_state.done,
            env_state.info["truncation"],
            env_state.info,
        )
        # next_obs, rewards, terminations, truncations, infos = envs.step(env_state, actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                break

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        # real_next_obs = next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]
        # rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        trajectory = get_rb_item_from_state(env_state, obs, actions)
        rb_state = rb.add(rb_state, trajectory)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # data = rb.sample(args.batch_size)
            data = rb.sample(rb_state, sample_key).experience

            qf1_state, qf1_loss_value, qf1_a_values = update_critic(
                actor_state,
                qf1_state,
                np.array(data["obs"]),
                np.array(data["actions"]),
                np.array(data["next_obs"]),
                np.array(data["rewards"].flatten()),
                np.array(data["dones"].flatten()),
            )
            if global_step % args.policy_frequency == 0:
                actor_state, qf1_state, actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    np.array(data["obs"]),
                )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                writer.add_scalar(
                    "losses/actor_loss", actor_loss_value.item(), global_step
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    # if args.save_model:
    #     model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    #     with open(model_path, "wb") as f:
    #         f.write(
    #             flax.serialization.to_bytes(
    #                 [
    #                     actor_state.params,
    #                     qf1_state.params,
    #                 ]
    #             )
    #         )
    #     print(f"model saved to {model_path}")
    # from cleanrl_utils.evals.ddpg_jax_eval import evaluate

    # episodic_returns = evaluate(
    #     model_path,
    #     make_env,
    #     args.env_id,
    #     eval_episodes=10,
    #     run_name=f"{run_name}-eval",
    #     Model=(Actor, QNetwork),
    #     exploration_noise=args.exploration_noise,
    # )
    # for idx, episodic_return in enumerate(episodic_returns):
    #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

    # if args.upload_model:
    #     from cleanrl_utils.huggingface import push_to_hub

    #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
    #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
    #     push_to_hub(
    #         args,
    #         episodic_returns,
    #         repo_id,
    #         "DDPG",
    #         f"runs/{run_name}",
    #         f"videos/{run_name}-eval",
    #     )

    # envs.close()
    writer.close()
