import time
import orbax.checkpoint as ocp

from evorl.distributed import unpmap
from evorl.types import State
from evorl.recorders import add_prefix

from evorl.algorithms.offpolicy_utils import skip_replay_buffer_state
from evorl.algorithms.td3 import TD3Workflow


class TD3WorkflowMod(TD3Workflow):
    """TD3Workflow with total_episode termination condition."""

    def learn(self, state: State) -> State:
        runtime = 0.0
        total_seconds = self.config.total_minutes * 60

        while runtime < total_seconds:
            start_time = time.perf_counter()
            train_metrics, state = self._multi_steps(state)
            train_time = time.perf_counter() - start_time
            runtime += train_time

            # current iteration
            iterations = unpmap(state.metrics.iterations, self.pmap_axis_name).tolist()
            workflow_metrics = state.metrics
            workflow_metrics_dict = workflow_metrics.to_local_dict()
            workflow_metrics_dict["train_runtime"] = runtime
            train_metrics = unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = unpmap(workflow_metrics, self.pmap_axis_name)
            self.recorder.write(train_metrics.to_local_dict(), iterations)
            self.recorder.write(workflow_metrics_dict, iterations)

            if iterations % self.config.eval_interval == 0 or runtime >= total_seconds:
                eval_metrics, state = self.evaluate(state)
                eval_metrics = unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iterations
                )

            saved_state = unpmap(state, self.pmap_axis_name)
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)
            self.checkpoint_manager.save(
                iterations,
                args=ocp.args.StandardSave(saved_state),
            )

        return state
