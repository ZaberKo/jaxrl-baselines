# Brax-blines

## Setup

This repo does not require cleanrl requirements, just directly install our `requirements.txt`

```shell
pip install -U -r requirements.txt
```

## Train

Example:

```shell
# Train one with certain random seed and config file without tracking
python train.py run_mode=normal seed=42 track=false --config-name=ddpg 


# Train multipul experiments for one config file with differenct random seeds (default 5 seeds)
python train.py run_mode=test track=false --config-name=ddpg

```
