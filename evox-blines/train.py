from evox import workflows, algorithms, problems
from evox.monitors import EvalMonitor

import jax
import jax.numpy as jnp


from utils import get_output_dir, metrics_todict, set_omegaconf_resolvers
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb


@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="config",
)
def train(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    output_dir = get_output_dir()

    train_fn = hydra.utils.get_method(config.algo.train_fn)

    wandb_name = f"evox-{config.algo.name}-{config.env_name}"
    wandb.init(
        project=config.wandb.project,
        name=wandb_name,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        tags=config.wandb.tags,
        dir=output_dir
    )
    
    try:
        train_fn(config)
    except BaseException as e:
        print(e)
        wandb.finish(1)

    wandb.finish()


if __name__ == "__main__":
    train()
