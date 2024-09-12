from brax import envs
import jax
import jax.numpy as jnp
from evox.utils import TreeAndVector
from evox import workflows, algorithms, problems
import wandb

from networks import MLPPolicy


def train(config):
    agent_key, workflow_key = jax.random.split(jax.random.PRNGKey(config.seed))

    env = envs.get_environment(env_name=config.env_name)

    action_dim = env.action_size
    obs_dim = env.observation_size

    print(f"{config.env_name}: obs_dim={obs_dim}, action_dim={action_dim}")

    model = MLPPolicy(action_dim=action_dim)
    params = model.init(agent_key, jnp.ones((1, obs_dim)))
    adapter = TreeAndVector(params)

    problem = problems.neuroevolution.Brax(
        policy=model.apply,
        env_name=config.env_name,
        max_episode_length=config.max_episode_length,
        num_episodes=config.algo.episodes_for_fitness,
    )

    algorithm = algorithms.CMAES(
        center_init=adapter.to_vector(params),
        init_stdev=config.algo.init_stdev,
        pop_size=config.algo.pop_size,
    )

    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        opt_direction="max",
        candidate_transforms=[adapter.batched_to_tree],
    )

    state = workflow.init(workflow_key)

    for i in range(config.algo.num_steps):
        train_info, state = workflow.step(state)
        episode_returns = train_info["fitness"] * workflow.opt_direction
        best_episode_return = jnp.max(episode_returns)
        

        iters = i + 1

        print(f"step {iters} best episode_return={best_episode_return}")
        wandb.log(
            dict(
                episode_returns=episode_returns,
                best_episode_return=best_episode_return,
                iters=iters,
            ),
            step=iters,
        )
