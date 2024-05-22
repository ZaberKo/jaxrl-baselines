import hydra
from omegaconf import DictConfig, OmegaConf
import os
import wandb

os.environ["WANDB_API_KEY"] = "5341b8b0aa7a3635f2ea7c71e238ce16b52dfd9b"
os.environ["WANDB_MODE"] = "offline"
os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(version_base=None, config_path="./config", config_name="c51")
def main(config: DictConfig):
    run_name = "test"
    # print(OmegaConf.to_yaml(config))
    wandb.init(
        project=config.wandb_project_name,
        entity=config.wandb_entity,
        name=run_name,
        config=OmegaConf.to_container(config, resolve=True),
        mode="offline"
    )
    wandb.log({"test":0})
    wandb.finish()
    print("finish")
if __name__ == "__main__":
    main()