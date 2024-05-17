import hydra
from omegaconf import DictConfig, OmegaConf

from agent.ddpg import main as ddpg

@hydra.main(version_base=None, config_path="./config", config_name="td3")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    ddpg(config)

    # TRY NOT TO MODIFY: seeding
   


if __name__ == "__main__":
    main()
