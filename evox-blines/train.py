import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
import time

import jax
import jax.numpy as jnp
from jax import config as jax_config

from utils import get_output_dir, set_omegaconf_resolvers

jax_config.update("jax_debug_nans", True)
set_omegaconf_resolvers()

@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="config",
)
def train(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    output_dir = get_output_dir()

    train_fn = hydra.utils.get_method(config.train_fn)

    wandb_name = f"evox-{config.name}-{config.env}"
    wandb_tags = list(config.wandb.tags)
    if len(wandb_tags) > 0:
        wandb_name += f"|{','.join(wandb_tags)}"
    wandb_tags.append("evox")
    wandb_tags.append(config.name)

    wandb.init(
        project=config.wandb.project,
        name=wandb_name,
        group=wandb_name,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        tags=wandb_tags,
        dir=output_dir
    )
    
    try:
        start_t = time.perf_counter()
        train_fn(config)
        end_t = time.perf_counter()
        print(f"Training took {end_t - start_t:.2f}/3600 hours")
    except Exception as e:
        raise e
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
