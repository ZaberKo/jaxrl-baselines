# Brax-blines

Example:
```shell
python train.py agent=sac/hopper training_config.num_evals=100


# train all
python train.py -m agent=ppo/ant,ppo/halfcheetah,ppo/humanoid,ppo/inverted_double_pendulum,ppo/inverted_pendulum,ppo/pusher,ppo/reacher,sac/hopper,sac/walker2d ++training_config.num_evals=100
python train.py agent=ppo/humanoidstandup training_config.num_evals=200
```