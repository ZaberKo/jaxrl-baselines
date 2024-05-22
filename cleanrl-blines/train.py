import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
import os

from agent.ddpg import main as ddpg
from agent.td3 import main as td3
from agent.dqn import main as dqn
from agent.ddpg_brax import main as ddpg_brax
from agent.td3_brax import main as td3_brax
from agent.c51 import main as c51

os.environ["WANDB_API_KEY"] = "5341b8b0aa7a3635f2ea7c71e238ce16b52dfd9b"
os.environ["WANDB_MODE"] = "offline"
os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(version_base=None, config_path="./config", config_name="dqn")
def test_main(config: DictConfig):
    seeds = [42, 3407, 114514, 7, 1, 2021, 31415, 999, 500, 1024]
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
        elif agent_type == 'c51':
            c51(config)
        elif agent_type == "ddpg_brax":
            ddpg_brax(config)
        elif agent_type == "td3_brax":
            td3_brax(config)
        else:
            raise ValueError("Unsupported agent type specified in the configuration!")

    # TRY NOT TO MODIFY: seeding


@hydra.main(version_base=None, config_path="./config", config_name="c51")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    agent_type = config.agent
    if agent_type == "dqn":
        dqn(config)
    elif agent_type == "td3":
        td3(config)
    elif agent_type == "ddpg":
        ddpg(config)
    elif agent_type == 'c51':
        c51(config)
    elif agent_type == "ddpg_brax":
        ddpg_brax(config)
    elif agent_type == "td3_brax":
        td3_brax(config)
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
