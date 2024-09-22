import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from utils import get_output_dir, set_omegaconf_resolvers

set_omegaconf_resolvers()


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    output_dir = get_output_dir()

    train_fn = hydra.utils.get_method(config.train_fn)

    run_name = f"cleanrl_{config.name}_{config.env_id}"

    wandb_tags = list(config.wandb.tags)
    if len(wandb_tags) > 0:
        wandb_name += f"|{','.join(wandb_tags)}"
    wandb_tags.append("evox")
    wandb_tags.append(config.name)

    wandb.init(
        project=config.wandb.project,
        config=OmegaConf.to_container(config, resolve=True),
        name=run_name,
        group=run_name,
        tags=wandb_tags,
        dir=output_dir,
    )

    train_fn(config)


if __name__ == "__main__":
    main()
