import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
import time

import jax
import jax.numpy as jnp
from evox import workflows, algorithms, problems
from evox.monitors import EvalMonitor

from utils import get_output_dir, set_omegaconf_resolvers


set_omegaconf_resolvers()

@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="config",
)
def train(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    output_dir = get_output_dir()

    train_fn = hydra.utils.get_method(config.algo.train_fn)

    wandb_name = f"evox-{config.algo.name}-{config.env}"
    wandb.init(
        project=config.wandb.project,
        name=wandb_name,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        tags=config.wandb.tags,
        dir=output_dir
    )
    
    try:
        start_t = time.perf_counter()
        train_fn(config)
        end_t = time.perf_counter()
        print(f"Training took {end_t - start_t:.2f}/3600 hours")
    except BaseException as e:
        print(e)
        wandb.finish(1)

    wandb.finish()


if __name__ == "__main__":
    train()
