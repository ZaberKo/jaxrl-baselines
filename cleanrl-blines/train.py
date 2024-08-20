import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
import os

from agent.ddpg import main as ddpg
from agent.td3 import main as td3
from agent.dqn import main as dqn

os.environ["WANDB_MODE"] = "offline"

@hydra.main(version_base=None, config_path="./config", config_name="ddpg")
def test_main(config: DictConfig):
    seeds = [42, 3407, 114514, 7, 1, 2021, 31415, 999, 500, 1024, 666]
    print(OmegaConf.to_yaml(config))
    for i in seeds:
        config.seed = i
        print("random seed={}".format(i))
        agent_type = config.agent
        if agent_type == "dqn":
            dqn(config)
        elif agent_type == "td3":
            td3(config)
        elif agent_type == "ddpg":
            ddpg(config)
        else:
            raise ValueError("Unsupported agent type specified in the configuration!")

    # TRY NOT TO MODIFY: seeding


@hydra.main(version_base=None, config_path="./config", config_name="td3")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    agent_type = config.agent
    if agent_type == "dqn":
        dqn(config)
    elif agent_type == "td3":
        td3(config)
    elif agent_type == "ddpg":
        ddpg(config)
    else:
        raise ValueError("Unsupported agent type specified in the configuration!")


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, required=True, help="Configuration name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser

if __name__ == "__main__":
    
    # main()
    test_main()
