import argparse
from absl import app

def main():
    parser = argparse.ArgumentParser(description="set wandb env and seed")

    parser.add_argument('agent_name', type=str, default='ppo')
    parser.add_argument('--env_name', type=str, default='brax:hopper')
    parser.add_argument('--num_steps', type=int, default=5_000_000)
    parser.add_argument('--eval_every', type=int, default=10_000)
    parser.add_argument('--evaluation_episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_project_name', type=str, default='acme-brax-baselines')
    parser.add_argument('--wandb_log_name', type=str, default='acme-brax-ppo-hopper_mean')
    parser.add_argument('--wandb_log_tag', type=str, default='acme')

    args = parser.parse_args()

    # app.run(run_ppo.train_agent)

    if args.agent_name == 'ppo':
        import run_ppo
        app.run(run_ppo.train_agent)
    elif args.agent_name == 'td3':
        import run_td3
        app.run(run_td3.train_agent)
    # elif args.agent_name == 'impala':
    #     run_impala()
    else:
        raise ValueError("Unsupported agent type specified in the configuration!")

if __name__ == '__main__':
    main()