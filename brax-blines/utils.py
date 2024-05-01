from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import yaml
from absl import logging
import jax.tree_util as jtu
import jax
import chex

def get_output_dir(default_path: str = './debug'):
    if HydraConfig.initialized():
        output_dir = Path(HydraConfig.get().run.dir).absolute()
    else:
        output_dir = Path(default_path).absolute()

    return output_dir


def metrics_todict(metrics):
    return jtu.tree_map(lambda x: x.tolist() if isinstance(x, chex.Array) else x, metrics)


def log_metrics(metrics):
    logging.info(
        "metrics:\n"+yaml.dump(metrics_todict(metrics), indent=2)
    )
