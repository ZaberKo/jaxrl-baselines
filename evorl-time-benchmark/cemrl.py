import time

import jax.tree_util as jtu
import orbax.checkpoint as ocp

from evorl.types import State
from evorl.recorders import get_1d_array_statistics, add_prefix
from evorl.algorithms.offpolicy_utils import skip_replay_buffer_state

from evorl.algorithms.erl.cemrl_td3.cemrl import (
    CEMRLWorkflow,
    get_std_statistics,
)


class CEMRLWorkflowMod(CEMRLWorkflow):
    def learn(self, state: State) -> State:
        runtime = 0.0
        total_seconds = self.config.total_minutes * 60

        while runtime < total_seconds:
            start_time = time.perf_counter()
            train_metrics, state = self.step(state)
            train_time = time.perf_counter() - start_time
            runtime += train_time

            iterations = state.metrics.iterations.tolist()
            workflow_metrics = state.metrics
            workflow_metrics_dict = workflow_metrics.to_local_dict()
            workflow_metrics_dict["train_runtime"] = runtime
            self.recorder.write(workflow_metrics_dict, iterations)

            train_metrics_dict = train_metrics.to_local_dict()
            train_metrics_dict["pop_episode_returns"] = get_1d_array_statistics(
                train_metrics_dict["pop_episode_returns"], histogram=True
            )

            train_metrics_dict["pop_episode_lengths"] = get_1d_array_statistics(
                train_metrics_dict["pop_episode_lengths"], histogram=True
            )

            if train_metrics_dict["rl_metrics"] is not None:
                train_metrics_dict["rl_metrics"]["raw_loss_dict"] = jtu.tree_map(
                    get_1d_array_statistics,
                    train_metrics_dict["rl_metrics"]["raw_loss_dict"],
                )
            self.recorder.write(train_metrics_dict, iterations)

            std_statistics = get_std_statistics(state.ec_opt_state.variance["params"])
            self.recorder.write({"ec/std": std_statistics}, iterations)

            if iterations % self.config.eval_interval == 0 or runtime >= total_seconds:
                eval_metrics, state = self.evaluate(state)

                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iterations
                )

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)

            self.checkpoint_manager.save(
                iterations, args=ocp.args.StandardSave(saved_state)
            )

        return state
