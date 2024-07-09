"""Example running IMPALA on brax env"""

import functools
from datetime import datetime
import hydra
import wandb
from omegaconf import OmegaConf, DictConfig
from utils import get_output_dir, metrics_todict, set_omegaconf_resolvers

import brax.v1.envs as v1_envs
from brax import envs

set_omegaconf_resolvers()

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def train(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    output_dir = get_output_dir()

    wandb.init(
        project=config.wandb.project,
        name=config.wandb.name,
        config=OmegaConf.to_container(config, resolve=True),
        tags=config.wandb.tags,
        dir=output_dir
    )

    try:
        
        train_fn = hydra.utils.get_method(config.train_fn)

        times = [datetime.now()]

        def wandb_progess_fn(env_steps, metrics):
            times.append(datetime.now())
            wandb.log(metrics_todict(metrics), env_steps)

        train_fn()
        
        print(f'time to jit: {times[1] - times[0]}')
        print(f'time to train: {times[-1] - times[1]}')
    except Exception as e:
        print(e)
        wandb.finish(1)
    
    wandb.finish()

if __name__ == '__main__':
    train()