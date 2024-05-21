import hydra
from omegaconf import DictConfig, OmegaConf
import argparse

from agent.ddpg import main as ddpg
from agent.td3 import main as td3
from agent.dqn import main as dqn
from agent.ddpg_brax import main as ddpg_brax
from agent.td3_brax import main as td3_brax

@hydra.main(version_base=None, config_path="./config", config_name="dqn")
def main(config: DictConfig):
    config.seed = args.seed
    print(OmegaConf.to_yaml(config))
    agent_type = config.agent
    if agent_type == "dqn":
        dqn(config)
    elif agent_type == "td3":
        td3(config)
    elif agent_type == "ddpg":
        ddpg(config)
    elif agent_type == "ddpg_brax":
        ddpg_brax(config)
    elif agent_type == "td3_brax":
        td3_brax(config)
    else:
        raise ValueError("Unsupported agent type specified in the configuration!")

    # TRY NOT TO MODIFY: seeding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main()
