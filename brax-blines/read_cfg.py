# %%
from omegaconf import OmegaConf
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
from pprint import pprint
import yaml

root_path = Path('configs')


ppo_10 = root_path/'ppo_10_million_steps.json'
ppo_500 = root_path/'ppo_500_million_steps.json'
sac_5 = root_path/'sac_5_million_steps.json'


# # %%
# with ppo_10.open('r') as f:
#     data = json.load(f)

# config_dict = defaultdict(list)
# config_list = [d['hyperparams'] for d in data]

# for cfg in config_list:
#     config_dict[cfg['env']].append(cfg)

# for env, cfgs in config_dict.items():
#     print(env)
#     pprint(cfgs[0])
# %%


def read_best_cfg(path):
    with path.open('r') as f:
        data = json.load(f)

    config_dict = defaultdict(list)
    config_list = [d['hyperparams'] for d in data]

    for cfg in config_list:
        config_dict[cfg['env']].append(cfg)

    for env, cfgs in config_dict.items():
        print(env, "#cfgs:", len(cfgs))
        # pprint(cfgs[0])
        print(yaml.dump(cfgs[0], indent=2))
        print()


# %%
read_best_cfg(ppo_10)

# %%
read_best_cfg(ppo_500)
# %%


def write_best_cfg(path, dst_path):
    with path.open('r') as f:
        data = json.load(f)

    if not dst_path.exists():
        dst_path.mkdir(parents=True, exist_ok=True)

    config_dict = defaultdict(list)
    config_list = [d['hyperparams'] for d in data]

    for cfg in config_list:
        config_dict[cfg['env']].append(cfg)

    for env, cfgs in config_dict.items():
        print(yaml.dump(cfgs[0], indent=2))
        cfg = OmegaConf.create(cfgs[0])
        env_name = cfg.pop('env')
        cfg.pop('learner')
        cfg.pop('eval_frequency')
        cfg.num_evals = 100
        cfg.num_timesteps = cfg.pop('total_env_steps')
        assert env == env_name

        new_cfg = OmegaConf.create()
        new_cfg.train_fn = 'agents.ppo.train.train'
        new_cfg.env_name = env_name
        new_cfg.training_config = cfg
        new_cfg.wandb = dict(name=f'ppo-{env_name}-10m')

        OmegaConf.save(new_cfg, dst_path/f"{env_name}.yaml")
        with (dst_path/f"{env_name}.yaml").open('r') as f:
            data = f.read()
        with (dst_path/f"{env_name}.yaml").open('w') as f:
            f.write('# @package _global_\n\n')
            f.write(data)


write_best_cfg(ppo_10, root_path/'agent'/'ppo-10m')
# %%


def write_best_cfg(path, dst_path):
    with path.open('r') as f:
        data = json.load(f)

    if not dst_path.exists():
        dst_path.mkdir(parents=True, exist_ok=True)

    config_dict = defaultdict(list)
    config_list = [d['hyperparams'] for d in data]

    for cfg in config_list:
        config_dict[cfg['env']].append(cfg)

    for env, cfgs in config_dict.items():
        print(yaml.dump(cfgs[0], indent=2))
        cfg = OmegaConf.create(cfgs[0])
        env_name = cfg.pop('env')
        cfg.pop('learner')
        cfg.pop('eval_frequency')
        cfg.num_evals = 200
        cfg.num_timesteps = cfg.pop('total_env_steps')
        assert env == env_name

        new_cfg = OmegaConf.create()
        new_cfg.train_fn = 'agents.ppo.train.train'
        new_cfg.env_name = env_name
        new_cfg.training_config = cfg
        new_cfg.wandb = dict(name=f'ppo-{env_name}-500m')

        OmegaConf.save(new_cfg, dst_path/f"{env_name}.yaml")

        with (dst_path/f"{env_name}.yaml").open('r') as f:
            data = f.read()
        with (dst_path/f"{env_name}.yaml").open('w') as f:
            f.write('# @package _global_\n\n')
            f.write(data)


write_best_cfg(ppo_500, root_path/'agent'/'ppo-500m')
# %%


def write_best_cfg(path, dst_path):
    with path.open('r') as f:
        data = json.load(f)

    if not dst_path.exists():
        dst_path.mkdir(parents=True, exist_ok=True)

    config_dict = defaultdict(list)
    config_list = [d['hyperparams'] for d in data]

    for cfg in config_list:
        config_dict[cfg['env']].append(cfg)

    for env, cfgs in config_dict.items():
        print(yaml.dump(cfgs[0], indent=2))
        cfg = OmegaConf.create(cfgs[0])
        env_name = cfg.pop('env')
        cfg.pop('learner')
        cfg.pop('eval_frequency')
        cfg.num_evals = 100
        cfg.num_timesteps = cfg.pop('total_env_steps')
        cfg.max_devices_per_host = 1
        assert env == env_name

        new_cfg = OmegaConf.create()
        new_cfg.train_fn = 'agents.sac.train.train'
        new_cfg.env_name = env_name
        new_cfg.training_config = cfg
        new_cfg.wandb = dict(name=f'sac-{env_name}-10m')

        OmegaConf.save(new_cfg, dst_path/f"{env_name}.yaml")
        with (dst_path/f"{env_name}.yaml").open('r') as f:
            data = f.read()
        with (dst_path/f"{env_name}.yaml").open('w') as f:
            f.write('# @package _global_\n\n')
            f.write(data)


write_best_cfg(sac_5, root_path/'agent'/'sac-5m')
# %%
