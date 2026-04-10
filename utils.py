"""utils.py - plotting helpers for RL experiments."""

import os
import numpy as np
import matplotlib.pyplot as plt


def smooth(rewards, window=50):
    """Rolling mean over `window` episodes."""
    if len(rewards) < window:
        return rewards
    return np.convolve(rewards, np.ones(window) / window, mode="valid")


def plot_rewards(histories, labels, title="Reward vs Episodes", save_path=None):
    """Plot smoothed reward curves for multiple runs."""
    plt.figure(figsize=(9, 4))
    for rewards, label in zip(histories, labels):
        smoothed = smooth(rewards)
        x = range(len(rewards) - len(smoothed), len(rewards))
        plt.plot(x, smoothed, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Total Reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_q_heatmap(agent, title="Q-value advantage (flap − noop)", save_path=None):
    """Heatmap of Q(s, flap) − Q(s, noop) over the (dx, dy) state grid."""
    if not agent.Q:
        print("Q-table is empty - train first.")
        return

    states = list(agent.Q.keys())
    xs = sorted(set(s[0] for s in states))
    ys = sorted(set(s[1] for s in states))

    grid = np.zeros((len(ys), len(xs)))
    for i, dy in enumerate(ys):
        for j, dx in enumerate(xs):
            s = (dx, dy)
            q = agent.Q.get(s, [0.0, 0.0])
            grid[i, j] = q[1] - q[0]  # advantage of flapping

    plt.figure(figsize=(8, 5))
    im = plt.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn")
    plt.colorbar(im, label="Q(flap) − Q(noop)")
    plt.xticks(range(len(xs)), xs, fontsize=7)
    plt.yticks(range(len(ys)), ys, fontsize=7)
    plt.xlabel("dx (horizontal distance to pipe gap)")
    plt.ylabel("dy (vertical distance to pipe gap)")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_param_sweep(results, param_name, title=None, save_path=None):
    """Plot reward curves for multiple hyperparameter values.

    results: list of (label, reward_history)
    """
    plt.figure(figsize=(9, 4))
    for label, rewards in results:
        s = smooth(rewards)
        x = range(len(rewards) - len(s), len(rewards))
        plt.plot(x, s, label=f"{param_name}={label}")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.title(title or f"Sensitivity to {param_name}")
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_generalization(configs, scores, title="Generalization across configs", save_path=None):
    """Bar chart comparing mean scores across environment configs.

    configs: list of str labels
    scores:  list of float mean scores
    """
    plt.figure(figsize=(8, 4))
    bars = plt.bar(configs, scores, color=["steelblue" if i == 0 else "coral" for i in range(len(configs))])
    plt.ylabel("Mean Score")
    plt.title(title)
    plt.xticks(rotation=15, ha="right")
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{score:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def evaluate(
    agent,
    env,
    n_episodes=100,
    show_progress=False,
    progress_desc="Evaluation",
    max_steps_per_episode=None,
):
    """Run agent greedily (eps=0) and return mean total reward and mean score."""
    from env_utils import get_state
    original_eps = agent.epsilon
    agent.epsilon = 0.0
    rewards, scores = [], []
    episode_iter = range(n_episodes)
    step_limit = max_steps_per_episode or getattr(env, "_max_episode_steps", None)

    if show_progress:
        from tqdm.auto import tqdm

        episode_iter = tqdm(
            episode_iter,
            desc=progress_desc,
            dynamic_ncols=True,
            smoothing=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    for _ in episode_iter:
        obs, info = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            action = agent.select_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if step_limit is not None and steps >= step_limit:
                done = True
            state = get_state(obs)
            total_reward += reward
        rewards.append(total_reward)
        scores.append(info.get("score", 0))

    agent.epsilon = original_eps
    return float(np.mean(rewards)), float(np.mean(scores))
