import brax
from brax import envs
import gymnasium as gym
from brax.envs.wrappers import gym as gym_wrapper

env_name = 'hopper'
num_envs = 1
episode_length = 1

env = envs.create(env_name, batch_size=num_envs, episode_length=episode_length)
env = gym_wrapper.VectorGymWrapper(env)
a = env.action_space

env.reset()
# action = torch.zeros(env.action_space.shape).to(device)
# env.step(action)
