import hydra
from omegaconf import DictConfig, OmegaConf

from agent.ddpg import main as ddpg
from agent.td3 import main as td3
from agent.dqn import main as dqn
from agent.ddpg_brax import main as ddpg_brax
from agent.td3_brax import main as td3_brax

@hydra.main(version_base=None, config_path="./config", config_name="dqn")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    dqn(config)

    # TRY NOT TO MODIFY: seeding


if __name__ == "__main__":
    main()
