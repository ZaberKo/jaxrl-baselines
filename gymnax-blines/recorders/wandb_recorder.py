import wandb
from .recorder import Recorder
from typing import Mapping, Any, Optional

class WandbRecorder(Recorder):
    def __init__(self, *, project, name, config, mode='disabled', **kwargs):
        wandb.init(
            project=project,
            name=name,
            config=config,
            mode=mode,
            **kwargs
        )

    def write(self, data: Mapping[str, Any], step: Optional[int] = None, step_name: Optional[str] = None) -> None:
        if step_name:
            wandb.log(data, step=step, commit=False)
            wandb.define_metric(step_name, step_metric=step_name)
            wandb.log(data, step=step, commit=True)
        else:
            wandb.log(data, step=step)

    def close(self):
        wandb.finish()
