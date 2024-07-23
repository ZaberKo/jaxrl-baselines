from absl import flags
from acme.agents.jax import ppo
import helpers
from absl import app
from acme.jax import experiments
import run_experiment
from acme.utils import lp_utils
import launchpad as lp
from utils import get_output_dir
import wandb
import os

os.environ["WANDB_MODE"] = "offline"
FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('env_name', 'brax:halfcheetah', 'What environment to run')
flags.DEFINE_string('wandb_project_name', 'acme-brax-baselines', 'What wandb project to log')
flags.DEFINE_string('wandb_log_name', 'acme-brax-ppo-halfcheetah_mean', 'The name of log')
flags.DEFINE_string('wandb_log_tag', 'acme', 'The tag of wandb log')
flags.DEFINE_integer('seed', 21, 'Random seed.')
flags.DEFINE_integer('num_steps', 5_000_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 10_000, 'How often to run evaluation.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')
flags.DEFINE_integer('num_distributed_actors', 8,
                     'Number of actors to use in the distributed setting.')

def build_experiment_config():
    suite, task = FLAGS.env_name.split(':', 1)

    config = ppo.PPOConfig(
        normalize_advantage=True,
        normalize_value=True,
        obs_normalization_fns_factory=ppo.build_mean_std_normalizer
    )
    ppo_builder = ppo.PPOBuilder(config)

    layer_sizes = (256,256,256)
    return experiments.ExperimentConfig(
        builder=ppo_builder,
        environment_factory=lambda seed: helpers.make_environment(suite, task),
        network_factory= lambda spec: ppo.make_networks(spec, layer_sizes),
        seed=FLAGS.seed,
        max_num_actor_steps=FLAGS.num_steps)

def train_agent(_):
    output_dir = get_output_dir()
    print(FLAGS.env_name)

    wandb.init(
        project=FLAGS.wandb_project_name,
        name=FLAGS.wandb_log_name,
        # config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        tags=FLAGS.wandb_log_tag,
        dir=output_dir
    )
    config = build_experiment_config()
    if FLAGS.run_distributed:
        program = experiments.make_distributed_experiment(
            experiment=config,
            num_actors=FLAGS.num_distributed_actors
        )
        lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
    else:
        run_experiment.run_experiment(
            experiment=config,
            eval_every=FLAGS.eval_every,
            num_eval_episodes=FLAGS.evaluation_episodes
        )
    
    wandb.finish(1)

if __name__ == '__main__':
    app.run(train_agent)
