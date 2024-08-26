# Brax-blines

Example:
```shell
# Train one with certain seed without tracking
python train.py run_mode=normal seed=42 track=false --config-name=ddpg 


# Train multipul experiments for one config with differenct random seed
python train.py run_mode=test seed=42 track=false --config-name=ddpg

```