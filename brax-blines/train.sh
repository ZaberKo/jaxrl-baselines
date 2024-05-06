python train.py -m agent=ppo/ant,ppo/halfcheetah,ppo/humanoid,ppo/inverted_double_pendulum,ppo/inverted_pendulum,ppo/pusher,ppo/reacher,sac/hopper,sac/walker2d ++training_config.num_evals=100 seed=0,1,2,3,4
python train.py -m agent=ppo/humanoidstandup training_config.num_evals=200 seed=0,1,2,3,4


python train.py -m agent=ppo-10m/ant,ppo-10m/fetch,ppo-10m/grasp,ppo-10m/halfcheetah,ppo-10m/humanoid,sac-5m/ant,sac-5m/fetch,sac-5m/grasp,sac-5m/halfcheetah,sac-5m/humanoid seed=0,1,2,3,4

python train.py -m agent=ppo-100m/ant,ppo-100m/fetch,ppo-100m/grasp,ppo-100m/halfcheetah,ppo-100m/humanoid seed=0,1,2,3,4