# Brax-blines

Example:
```shell
# Train one with certain random seed and config file without tracking
python train.py run_mode=normal seed=42 track=false --config-name=ddpg 


# Train multipul experiments for one config file with differenct random seeds (default 5 seeds)
python train.py run_mode=test track=false --config-name=ddpg

```