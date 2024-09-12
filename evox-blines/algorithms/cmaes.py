from brax import envs
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from evox.utils import TreeAndVector
from evox import workflows, algorithms, problems
import numpy as np
import wandb

from networks import MLPPolicy
from utils import metrics_todict, get_1d_array_statistics
from evaluator import BraxEvaluator

from omegaconf import OmegaConf


def train(config):
    key, agent_key, workflow_key = jax.random.split(jax.random.PRNGKey(config.seed), 3)

    env = envs.get_environment(env_name=config.env)

    action_dim = env.action_size
    obs_dim = env.observation_size

    print(f"{config.env}: obs_dim={obs_dim}, action_dim={action_dim}")

    model = MLPPolicy(
        action_dim=action_dim,
        hidden_layer_sizes=tuple(config.policy_network.hidden_layer_sizes),
    )
    params = model.init(agent_key, jnp.ones((1, obs_dim)))
    adapter = TreeAndVector(params)

    problem = problems.neuroevolution.Brax(
        policy=model.apply,
        env_name=config.env,
        max_episode_length=config.max_episode_length,
        num_episodes=config.episodes_for_fitness,
    )

    pop_center = adapter.to_vector(params)

    print(f"num_params={pop_center.shape[0]}")

    num_elites = config.num_elites
    recombination_weights = jnp.log(num_elites + 0.5) - jnp.log(
        jnp.arange(1, num_elites + 1)
    )
    recombination_weights /= jnp.sum(recombination_weights)

    algorithm = algorithms.CMAES(
        center_init=pop_center,
        init_stdev=config.init_stdev,
        pop_size=config.pop_size,
        recombination_weights=recombination_weights,
    )

    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        opt_direction="max",
        candidate_transforms=[adapter.batched_to_tree],
    )

    evaluator = BraxEvaluator(
        policy=model.apply,
        env_name=config.env,
        max_episode_length=config.max_episode_length,
        num_episodes=config.eval_episodes,
    )
    evaluator.evaluate = jax.jit(evaluator.evaluate)

    state = workflow.init(workflow_key)

    print("Starting training:")
    total_sampled_episodes = 0
    global_best_episode_return = jnp.finfo(jnp.float32).min
    global_best_weights = None
    eval_global_best_flag = False

    for i in range(config.num_steps):
        train_info, state = workflow.step(state)

        algo_state = state.get_child_state("algorithm")
        episode_returns = train_info["fitness"] * workflow.opt_direction

        best_index = jnp.argmax(episode_returns)
        best_episode_return = episode_returns[best_index]
        if best_episode_return > global_best_episode_return:
            global_best_weights = algo_state.population[
                best_index
            ]
            global_best_episode_return = best_episode_return
            eval_global_best_flag = True
        else:
            eval_global_best_flag = False

        iters = i + 1

        total_sampled_episodes += config.episodes_for_fitness * config.pop_size

        metrics = dict(
            episode_returns=get_1d_array_statistics(episode_returns, histogram=True),
            best_episode_return=best_episode_return,
            global_best_episode_return=global_best_episode_return,
            sampled_episodes=total_sampled_episodes,
            iters=iters,
        )
       
        metrics["eval/sigma"] = algo_state.sigma.tolist()
        diag_cov = adapter.to_tree(jnp.diagonal(state.C))
        metrics["eval/std"] = get_std_statistics(diag_cov)

        print(f"step {iters}:")
        print(f"best_episode_return={best_episode_return:.2f}")
        print(f"global_best_episode_return={global_best_episode_return:.2f}")
        print(f"CMA-ES sigma={metrics['eval/sigma']}")

        if iters % config.eval_interval == 0:
            key, center_key = jax.random.split(key)
            episode_returns = evaluator.evaluate(
                adapter.to_tree(algo_state.mean), center_key
            )
            episode_returns_stats = dict(
                mean=jnp.mean(episode_returns).tolist(),
                min=jnp.min(episode_returns).tolist(),
                max=jnp.max(episode_returns).tolist(),
            )

            metrics["eval/pop_center_episode_return"] = episode_returns_stats
            print("+" * 20)
            print(f"pop_center_episode_return:")
            yaml_print(jtu.tree_map(lambda x: f"{x:.2f}", episode_returns_stats))

        if eval_global_best_flag:
            key, best_key = jax.random.split(key)
            episode_returns = evaluator.evaluate(
                adapter.to_tree(global_best_weights), best_key
            )
            episode_returns_stats = dict(
                mean=jnp.mean(episode_returns).tolist(),
                min=jnp.min(episode_returns).tolist(),
                max=jnp.max(episode_returns).tolist(),
            )

            metrics["eval/global_best_episode_return"] = episode_returns_stats
            print("+" * 20)
            print(f"global_best_episode_return:")
            yaml_print((jtu.tree_map(lambda x: f"{x:.2f}", episode_returns_stats)))

        wandb.log(
            metrics_todict(metrics),
            step=iters,
        )
        print("=" * 20)


def yaml_print(data):
    print(OmegaConf.to_yaml(data))


def get_std_statistics(variance):
    def _get_stats(x):

        x = jnp.sqrt(x)
        return dict(
            min=jnp.min(x).tolist(),
            max=jnp.max(x).tolist(),
            mean=jnp.mean(x).tolist(),
        )

    return jtu.tree_map(_get_stats, variance)
