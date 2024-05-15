import os
import random
import time
from dataclasses import dataclass

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import brax
from brax import envs
import hydra
from omegaconf import DictConfig, OmegaConf

from agent.ddpg import main as ddpg

@hydra.main(version_base=None, config_path="./config", config_name="ddpg")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    ddpg(config)

    # TRY NOT TO MODIFY: seeding
   


if __name__ == "__main__":
    main()
