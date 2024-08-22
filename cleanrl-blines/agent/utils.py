import brax.envs
from brax import envs
import brax.envs.wrappers
import flashbax
import jax.numpy as jnp
import flashbax
import numpy as np
import brax
import gymnasium as gym
import jax
from functools import partial


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

    average_reward = np.mean(np.array(total_rewards))
    return average_reward


def box_stanslate(brax_space):
    if isinstance(brax_space, brax.physics.config.Box):
        # 创建一个Gymnasium的Box空间
        low = brax_space.low
        high = brax_space.high
        return gym.spaces.Box(low=low, high=high, dtype=float)
    else:
        raise NotImplementedError("Unsupported space type")


class AttrDict:
    def __init__(self, d=None):
        if d is None:
            d = {}
        # 保证所有传入的字典项都可以通过属性访问
        self.__dict__.update(d)

    def __getattr__(self, key):
        # 如果试图访问的属性不存在，抛出AttributeError
        if key not in self.__dict__:
            raise AttributeError(f"No such attribute: {key}")
        return self.__dict__[key]

    def __setattr__(self, key, value):
        # 允许属性赋值
        self.__dict__[key] = value

    def __repr__(self):
        # 定制对象的打印信息，方便调试
        return str(self.__ddict__)


class Evaluator_for_gymnasium:
    def __init__(self, env, num_envs, seed):
        # 初始化并行环境
        self.env = env
        self.num_envs = num_envs
        self.key = jax.random.PRNGKey(seed)

    def evaluate(self, actor, actor_state):
        key, env_key = jax.random.split(self.key)
        env_seed = int(jax.random.randint(env_key, (), 0, 2**15))
        obs, _ = self.env.reset(seed=env_seed)
        rewards = []
        lengths = []
        self.key = key
        actor_apply = jax.jit(actor.apply)  # JIT 编译

        while len(rewards) < self.num_envs:
            # 计算动作
            actions = actor_apply(actor_state.params, obs)
            actions = jax.device_get(actions)
            # 执行动作
            next_obs, _, _, _, infos = self.env.step(actions)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        rewards.append(info["episode"]["r"])
                        lengths.append(info["episode"]["l"])

            obs = next_obs  # 更新当前的状态

        rewards = jnp.array(rewards)
        lengths = jnp.array(lengths)

        return jnp.mean(rewards), jnp.mean(lengths)


class Evaluator_for_brax:
    def __init__(self, env_id, seed=0, num_envs=10, num_episodes=1000):
        # 创建并行环境
        self.env = brax.envs.create(
            env_id, batch_size=num_envs, episode_length=num_episodes
        )
        self.num_envs = num_envs
        self.key = jax.random.PRNGKey(seed)

    @partial(jax.jit, static_argnums=(0, 2))
    def _evaluate(self, env_key, actor_apply, actor_state):
        # 初始化环境和评估器状态
        env_state = self.env.reset(rng=env_key)
        init_vals = (
            env_state,
            jnp.zeros(self.num_envs),
            jnp.zeros(self.num_envs),
            jnp.zeros(self.num_envs, dtype=bool),
            0,
        )

        # 定义条件函数
        def cond_fn(vals):
            _, _, _, done, _ = vals
            return ~done.all()

        # 定义循环主体函数
        def body_fn(vals):
            env_state, total_rewards, episode_lengths, done, step_count = vals
            action = actor_apply(actor_state.params, env_state.obs)
            env_state = self.env.step(env_state, action)
            total_rewards = total_rewards + env_state.reward * (~done)
            done = jnp.logical_or(done, env_state.done)
            episode_lengths = episode_lengths + (~done)

            return (
                env_state,
                total_rewards,
                episode_lengths,
                done,
                step_count + 1,
            )

        # 执行 JAX 优化的 while 循环
        final_vals = jax.lax.while_loop(cond_fn, body_fn, init_vals)

        # 计算平均奖励和步长
        _, total_rewards, episode_lengths, _, _ = final_vals

        # 计算平均奖励和步长
        average_reward = jnp.mean(total_rewards)
        average_length = jnp.mean(episode_lengths)

        return average_reward, average_length

    def evaluate(self, actor, actor_state):
        key, env_key = jax.random.split(self.key)
        self.key = key
        average_reward, average_length = self._evaluate(
            env_key, actor.apply, actor_state
        )
        return average_reward, average_length
