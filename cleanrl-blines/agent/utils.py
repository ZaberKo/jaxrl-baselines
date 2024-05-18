import flashbax
from jax import jit
import jax.numpy as jnp
import flashbax
import numpy as np

def get_rb_item(obs, action, reward, next_obs, done, truncation):
    item = dict(
        obs=obs,
        actions=action,
        rewards=reward,
        next_obs=next_obs,
        dones=done,
        truncation=truncation,
    )
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


def test_actor_performance(envs, env_key, actor, actor_state, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        # Reset the environment for a new episode
        env_state = envs.reset(env_key)
        obs = env_state.obs
        done = False
        episode_reward = 0.0
        step_count = 0
        while not done and step_count < 1000:
            # Generate action from the actor network
            action = actor.apply(actor_state.params, obs)
            # Execute action in the environment
            env_state = envs.step(env_state, action)
            obs = env_state.obs
            reward = (0.99**step_count) * env_state.reward
            done = env_state.done
            step_count += 1
            # Accumulate rewards for the episode
            episode_reward += reward.sum()  # sum rewards if environment is vectorized

        total_rewards.append(episode_reward)

    average_reward =np.mean(np.array(total_rewards))
    return average_reward
