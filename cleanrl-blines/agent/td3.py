# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_action_jaxpy
import random
import time

import flax
import flax.linen as nn
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from brax import envs as brax_envs
from tensorboardX import SummaryWriter
from omegaconf import DictConfig
import warnings
from utils import (
    get_rb_item,
    get_rb_item_from_state,
    replayer_buffer,
    test_actor_performance,
)

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


def main(args: DictConfig) -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="brax.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="flashbax.*")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    key, actor_key, qf1_key, qf2_key, env_key = jax.random.split(key, 5)

    # env setup
    envs = brax_envs.create(env_name=args.env_name, batch_size=args.num_envs, )

    # max_action = float(envs.single_action_space.high[0])
    rb, rb_state = replayer_buffer(args, envs)
    env_state = envs.reset(env_key)
    obs = env_state.obs
    action_space = envs.sys.actuator.ctrl_range
    action_low = action_space[0, 0]
    action_high = action_space[0, 1]
    max_action = float(action_high)

    actor = Actor(
        action_dim=np.prod(envs.action_size),
        action_scale=jnp.array((action_high - action_low) / 2.0),
        action_bias=jnp.array((action_high + action_low) / 2.0),
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
    qf2_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf2_key, obs, jnp.ones((1, envs.action_size))),
        target_params=qf.init(qf2_key, obs, jnp.ones((1, envs.action_size))),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        key: jnp.ndarray,
    ):
        # TODO Maybe pre-generate a lot of random keys
        # also check https://jax.readthedocs.io/en/latest/jax.random.html
        key, noise_key = jax.random.split(key, 2)
        clipped_noise = (
            jnp.clip(
                (jax.random.normal(noise_key, actions.shape) * args.policy_noise),
                -args.noise_clip,
                args.noise_clip,
            )
            * actor.action_scale
        )
        next_state_actions = jnp.clip(
            actor.apply(actor_state.target_params, next_observations) + clipped_noise,
            envs.single_action_space.low,
            envs.single_action_space.high,
        )
        qf1_next_target = qf.apply(
            qf1_state.target_params, next_observations, next_state_actions
        ).reshape(-1)
        qf2_next_target = qf.apply(
            qf2_state.target_params, next_observations, next_state_actions
        ).reshape(-1)
        min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
        next_q_value = (
            rewards + (1 - terminations) * args.gamma * (min_qf_next_target)
        ).reshape(-1)

        def mse_loss(params):
            qf_a_values = qf.apply(params, observations, actions).squeeze()
            return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(
            mse_loss, has_aux=True
        )(qf1_state.params)
        (qf2_loss_value, qf2_a_values), grads2 = jax.value_and_grad(
            mse_loss, has_aux=True
        )(qf2_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads1)
        qf2_state = qf2_state.apply_gradients(grads=grads2)

        return (
            (qf1_state, qf2_state),
            (qf1_loss_value, qf2_loss_value),
            (qf1_a_values, qf2_a_values),
            key,
        )

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
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
        qf2_state = qf2_state.replace(
            target_params=optax.incremental_update(
                qf2_state.params, qf2_state.target_params, args.tau
            )
        )
        return actor_state, (qf1_state, qf2_state), actor_loss_value

    envs.step = jit(envs.step)
    start_time = time.time()
    for global_step in range(args.total_timesteps):
        key, loop_key, sample_key = jax.random.split(key, 3)
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
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
                            0,
                            max_action * args.exploration_noise,
                            size=envs.action_size,
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
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        trajectory = get_rb_item_from_state(env_state, obs, actions)
        rb_state = rb.add(rb_state, trajectory)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # data = rb.sample(args.batch_size)
            data = rb.sample(rb_state, sample_key).experience
            (
                (qf1_state, qf2_state),
                (qf1_loss_value, qf2_loss_value),
                (qf1_a_values, qf2_a_values),
                key,
            ) = update_critic(
                actor_state,
                qf1_state,
                qf2_state,
                np.array(data["obs"]),
                np.array(data["actions"]),
                np.array(data["next_obs"]),
                np.array(data["rewards"].flatten()),
                np.array(data["dones"].flatten()),
                key,
            )

            if global_step % args.policy_frequency == 0:
                actor_state, (qf1_state, qf2_state), actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    qf2_state,
                    np.array(data["obs"]),
                )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss_value.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.item(), global_step)
                writer.add_scalar(
                    "losses/actor_loss", actor_loss_value.item(), global_step
                )
                average_reward = test_actor_performance(
                    envs, env_key, actor, actor_state
                )
                writer.add_scalar("test/average_reward", average_reward, global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        actor_state.params,
                        qf1_state.params,
                        qf2_state.params,
                    ]
                )
            )
        print(f"model saved to {model_path}")
        # from cleanrl_utils.evals.td3_jax_eval import evaluate

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
        #         "TD3",
        #         f"runs/{run_name}",
        #         f"videos/{run_name}-eval",
        #     )
    # envs.close()
    writer.close()
