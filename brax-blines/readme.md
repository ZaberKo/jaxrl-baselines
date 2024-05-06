# Brax-blines

Example:
```shell
python train.py agent=sac/hopper training_config.num_evals=100


# Train all
python train.py -m agent=ppo/ant,ppo/halfcheetah,ppo/humanoid,ppo/inverted_double_pendulum,ppo/inverted_pendulum,ppo/pusher,ppo/reacher,sac/hopper,sac/walker2d ++training_config.num_evals=100
python train.py agent=ppo/humanoidstandup training_config.num_evals=200


# Train one with mulitple seed
python train.py -m agent=ppo/humanoidstandup training_config.num_evals=200 seed=0,1,2,3,4

```