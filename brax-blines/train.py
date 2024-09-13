import functools

from datetime import datetime
import brax.v1.envs as v1_envs

from brax import envs
from utils import get_output_dir, metrics_todict, set_omegaconf_resolvers
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb

set_omegaconf_resolvers()


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def train(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    output_dir = get_output_dir()

    wandb_name = config.wandb.name
    wandb_tags = list(config.wandb.tags)
    if len(wandb_tags) > 0:
        wandb_name += f"|{','.join(wandb_tags)}"
    wandb_tags.append("brax")

    wandb.init(
        project=config.wandb.project,
        name=wandb_name,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        tags=wandb_tags,
        dir=output_dir,
    )

    try:
        env_name = config.env_name

        if env_name in envs._envs:
            # prefer v2 envs
            env = envs.get_environment(env_name)
        elif env_name in v1_envs._envs:
            env = v1_envs.get_environment(env_name, legacy_spring=True)
        else:
            raise ValueError(f"Unknown environment {env_name}")

        train_fn = hydra.utils.get_method(config.train_fn)

        train_fn = functools.partial(train_fn, **config.training_config)

        times = [datetime.now()]

        def wandb_progess_fn(env_steps, metrics):
            times.append(datetime.now())
            wandb.log(metrics_todict(metrics), env_steps)
            # pprint(metrics)

        make_inference_fn, params, metrics = train_fn(
            environment=env, progress_fn=wandb_progess_fn
        )

        print(f"time to jit: {times[1] - times[0]}")
        print(f"time to train: {times[-1] - times[1]}")
    except Exception as e:
        raise e
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
