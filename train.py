"""train.py - Train and compare MC and Sarsa(lambda) agents on TextFlappyBird-v0.

Run:  python train.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from env_utils import make_env
from agents.mc_agent import MCAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from utils import plot_rewards, plot_q_heatmap, evaluate

# Default config
ENV_KWARGS = dict(height=15, width=20, pipe_gap=4)
N_EPISODES = 5_000


def run_mc(n_episodes=N_EPISODES, epsilon=0.1, gamma=0.99, **env_kwargs):
    env = make_env(**{**ENV_KWARGS, **env_kwargs})
    agent = MCAgent(epsilon=epsilon, gamma=gamma)
    rewards = agent.train(env, n_episodes=n_episodes)
    env.close()
    return agent, rewards


def run_sarsa(n_episodes=N_EPISODES, epsilon=0.1, alpha=0.1, gamma=0.99,
              lambda_=0.8, **env_kwargs):
    env = make_env(**{**ENV_KWARGS, **env_kwargs})
    agent = SarsaLambdaAgent(epsilon=epsilon, alpha=alpha,
                              gamma=gamma, lambda_=lambda_)
    rewards = agent.train(env, n_episodes=n_episodes)
    env.close()
    return agent, rewards


def compare():
    """Train both agents and produce comparison plots."""
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")

    print("Training MC agent...")
    mc_agent, mc_rewards = run_mc()
    print("Training Sarsa(lambda) agent...")
    sarsa_agent, sarsa_rewards = run_sarsa()

    # Reward curves
    plot_rewards(
        [mc_rewards, sarsa_rewards],
        ["MC", "Sarsa(lambda)"],
        title="Reward vs Episodes - MC vs Sarsa(lambda)",
        save_path=os.path.join(fig_dir, "fig_reward_curves.png"),
    )

    # Q heatmaps
    plot_q_heatmap(
        mc_agent,
        title="MC: Q(flap) - Q(noop)",
        save_path=os.path.join(fig_dir, "fig_q_heatmap_mc.png"),
    )
    plot_q_heatmap(
        sarsa_agent,
        title="Sarsa(lambda): Q(flap) - Q(noop)",
        save_path=os.path.join(fig_dir, "fig_q_heatmap_sarsa.png"),
    )

    # Evaluation
    eval_env_mc = make_env(**ENV_KWARGS)
    eval_env_sarsa = make_env(**ENV_KWARGS)
    mc_r, mc_s = evaluate(mc_agent, eval_env_mc)
    sarsa_r, sarsa_s = evaluate(sarsa_agent, eval_env_sarsa)
    eval_env_mc.close()
    eval_env_sarsa.close()

    print(f"\nMC       mean reward: {mc_r:.2f}, mean score: {mc_s:.2f}")
    print(f"Sarsa(lambda) mean reward: {sarsa_r:.2f}, mean score: {sarsa_s:.2f}")

    return mc_agent, sarsa_agent, mc_rewards, sarsa_rewards


if __name__ == "__main__":
    compare()
