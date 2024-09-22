# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_action_jaxpy

from tqdm import trange
from functools import partial

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState

from brax_envs import make_env
from replay_buffer import ReplayBuffer
from utils import BraxEvaluator, yaml_print


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: int
    action_scale: jax.Array
    action_bias: jax.Array

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


def main(config):

    # TRY NOT TO MODIFY: seeding
    # Note: we ensure all ops are on JAX.
    # random.seed(config.seed)
    # np.random.seed(config.seed)
    key = jax.random.PRNGKey(config.seed)
    key, rb_key, actor_key, qf1_key, qf2_key = jax.random.split(key, 5)

    # env setup
    # envs = gym.vector.SyncVectorEnv([make_env(config.env_id, config.seed, 0, config.capture_video, run_name)])
    envs = make_env(config.env_id, config.num_envs, config.seed)
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    # envs.single_observation_space.dtype = np.float32

    rb = ReplayBuffer(
        config.buffer_size,
        config.batch_size,
        envs.single_observation_space,
        envs.single_action_space,
        rb_key,
        enable_jit=True,
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=config.seed)

    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=config.learning_rate),
    )
    qf = QNetwork()
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf1_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=config.learning_rate),
    )
    qf2_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf2_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf2_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=config.learning_rate),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    evaluator = BraxEvaluator(
        config.env_id,
        actor.apply,
        config.eval_episodes,
    )

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        observations: jax.Array,
        actions: jax.Array,
        next_observations: jax.Array,
        rewards: jax.Array,
        terminations: jax.Array,
        key: jax.Array,
    ):
        # TODO Maybe pre-generate a lot of random keys
        # also check https://jax.readthedocs.io/en/latest/jax.random.html
        key, noise_key = jax.random.split(key, 2)
        clipped_noise = (
            jnp.clip(
                (jax.random.normal(noise_key, actions.shape) * config.policy_noise),
                -config.noise_clip,
                config.noise_clip,
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
            rewards + (1 - terminations) * config.gamma * (min_qf_next_target)
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
        observations: jax.Array,
    ):
        def actor_loss(params):
            return -qf.apply(
                qf1_state.params, observations, actor.apply(params, observations)
            ).mean()

        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(
                actor_state.params, actor_state.target_params, config.tau
            )
        )

        qf1_state = qf1_state.replace(
            target_params=optax.incremental_update(
                qf1_state.params, qf1_state.target_params, config.tau
            )
        )
        qf2_state = qf2_state.replace(
            target_params=optax.incremental_update(
                qf2_state.params, qf2_state.target_params, config.tau
            )
        )
        return actor_state, (qf1_state, qf2_state), actor_loss_value

    num_iters = config.total_timesteps // config.num_envs

    if config.progress_bar:
        _range = partial(trange, desc="global steps")
    else:
        _range = range

    for global_step in _range(config.total_timesteps // config.num_envs):
        # ALGO LOGIC: put action logic here

        sampled_timesteps = (global_step + 1) * config.num_envs

        key, action_key = jax.random.split(key)
        if sampled_timesteps <= config.learning_starts:
            actions = jax.random.uniform(
                action_key,
                (envs.num_envs,) + envs.single_action_space.shape,
                minval=envs.single_action_space.low,
                maxval=envs.single_action_space.high,
            )
        else:
            actions = actor.apply(actor_state.params, obs)
            actions += (
                jax.random.normal(action_key, actions.shape)
                * max_action
                * config.exploration_noise
            )
            actions = jnp.clip(
                actions,
                min=envs.single_action_space.low,
                max=envs.single_action_space.high,
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        # [num_envs, ...]
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            train_episodic_return_list = []
            train_episodic_length_list = []
            for info in infos["final_info"]:
                if info is not None:
                    train_episodic_return_list.append(info["episode"]["r"])
                    train_episodic_length_list.append(info["episode"]["l"])

            wandb.log(
                {
                    "training/episodic_return": np.mean(train_episodic_return_list),
                    "training/episodic_length": np.mean(train_episodic_return_list),
                    "sampled_timesteps": sampled_timesteps,
                    "global_step": global_step,
                },
                step=global_step,
            )

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        # real_next_obs = next_obs.copy()

        truncations = truncations.astype(jnp.bool)
        _real_next_obs = []
        for idx, trunc in enumerate(truncations):
            if trunc:
                # real_next_obs[idx] = infos["final_observation"][idx]
                _real_next_obs.append(infos["final_observation"][idx])
        if len(_real_next_obs) > 0:
            real_next_obs = next_obs.at[truncations].set(jnp.stack(_real_next_obs))
        else:
            real_next_obs = next_obs

        rb.add(obs, real_next_obs, actions, rewards, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if sampled_timesteps > config.learning_starts:
            # data = rb.sample(config.batch_size)
            data = rb.sample()

            (
                (qf1_state, qf2_state),
                (qf1_loss_value, qf2_loss_value),
                (qf1_a_values, qf2_a_values),
                key,
            ) = update_critic(
                actor_state,
                qf1_state,
                qf2_state,
                data.observations,
                data.actions,
                data.next_observations,
                data.rewards.flatten(),
                data.dones.flatten(),
                key,
            )

            if global_step % config.policy_frequency == 0:
                actor_state, (qf1_state, qf2_state), actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    qf2_state,
                    data.observations,
                )

            if (global_step + 1) % config.eval_freq == 0:
                key, eval_key = jax.random.split(key)
                episode_return, episode_length = evaluator.evaluate(
                    actor_state, eval_key
                )
                data = {
                    "training/qf1_loss": qf1_loss_value.item(),
                    "training/qf2_loss": qf2_loss_value.item(),
                    "training/qf1": qf1_a_values.item(),
                    "training/qf2": qf2_a_values.item(),
                    "training/actor_loss": actor_loss_value.item(),
                    "evalution/episodic_return": episode_return.mean().item(),
                    "evalution/episodic_length": episode_length.mean().item(),
                    "sampled_timesteps": sampled_timesteps,
                    "global_step": global_step,
                }
                yaml_print(data)
                wandb.log(data, step=global_step)
