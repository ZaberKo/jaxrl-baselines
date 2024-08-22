import hydra
from omegaconf import DictConfig, OmegaConf
import os

from agent.ddpg import main as ddpg
from agent.td3 import main as td3
from agent.dqn import main as dqn

os.environ["WANDB_MODE"] = "offline"

@hydra.main(version_base=None, config_path="./config", config_name="td3")
def main(config: DictConfig):
    # 打印当前配置
    print(OmegaConf.to_yaml(config))

    if config.run_mode == "normal":
        run_normal(config)
    elif config.run_mode == "test":
        run_test(config)
    else:
        raise ValueError(f"Unknown run_mode: {config.run_mode}")

def run_normal(config: DictConfig):
    # 根据配置中的种子进行实验
    agent_type = config.agent
    if agent_type == "dqn":
        dqn(config)
    elif agent_type == "td3":
        td3(config)
    elif agent_type == "ddpg":
        ddpg(config)
    else:
        raise ValueError("Unsupported agent type specified in the configuration!")

def run_test(config: DictConfig):
    seeds = [42, 3407, 114514, 7, 1, 2021, 31415, 999, 500, 1024, 666]
    for i in seeds:
        config.seed = i
        print(f"Running test with random seed={i}")
        agent_type = config.agent
        if agent_type == "dqn":
            dqn(config)
        elif agent_type == "td3":
            td3(config)
        elif agent_type == "ddpg":
            ddpg(config)
        else:
            raise ValueError("Unsupported agent type specified in the configuration!")

if __name__ == "__main__":
    main()
