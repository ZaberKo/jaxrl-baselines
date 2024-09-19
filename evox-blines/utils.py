from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import re

import jax.tree_util as jtu
import jax.numpy as jnp
import chex

import wandb

def set_omegaconf_resolvers():
    OmegaConf.register_new_resolver(
        "sanitize_dirname", lambda path: re.sub(r"/", "_", path)
    )


def get_output_dir(default_path: str = "./debug"):
    if HydraConfig.initialized():
        output_dir = Path(HydraConfig.get().runtime.output_dir).absolute()
    else:
        output_dir = Path(default_path).absolute()

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    return output_dir


def metrics_todict(metrics):
    return jtu.tree_map(
        lambda x: x.tolist() if isinstance(x, chex.Array) else x, metrics
    )


def get_1d_array_statistics(data, histogram=False):
    res = dict(
        min=jnp.min(data).tolist(),
        max=jnp.max(data).tolist(),
        mean=jnp.mean(data).tolist(),
    )

    res["val"] = np.array(data)
    if histogram:
        res["hist"] = wandb.Histogram(res["val"])

    return res
