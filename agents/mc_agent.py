"""mc_agent.py - First-Visit Monte Carlo Control with eps-greedy policy."""

import random
from collections import defaultdict

from env_utils import get_state


class MCAgent:
    """First-visit MC control agent using a tabular Q-table."""

    def __init__(self, n_actions=2, epsilon=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma

        # Q[state][action] = estimated action-value
        self.Q = defaultdict(lambda: [0.0] * n_actions)
        # returns[state][action] = list of observed returns (for averaging)
        self.returns = defaultdict(lambda: [[] for _ in range(n_actions)])

    def select_action(self, state):
        """eps-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(max(range(self.n_actions), key=lambda a: self.Q[state][a]))

    def _collect_episode(self, env, max_steps_per_episode=None):
        """Run one full episode and return list of (state, action, reward)."""
        obs, _ = env.reset()
        state = get_state(obs)
        trajectory = []
        done = False
        steps = 0
        step_limit = max_steps_per_episode or getattr(env, "_max_episode_steps", None)

        while not done:
            action = self.select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            if step_limit is not None and steps >= step_limit:
                done = True
            next_state = get_state(obs)
            trajectory.append((state, action, reward))
            state = next_state

        return trajectory

    def _update(self, trajectory):
        """First-visit MC update: compute returns and average them."""
        first_visit_idx = {}
        for idx, (state, action, _) in enumerate(trajectory):
            first_visit_idx.setdefault((state, action), idx)

        G = 0.0
        for idx in range(len(trajectory) - 1, -1, -1):
            state, action, reward = trajectory[idx]
            G = reward + self.gamma * G
            if first_visit_idx[(state, action)] == idx:
                self.returns[state][action].append(G)
                self.Q[state][action] = sum(self.returns[state][action]) / len(
                    self.returns[state][action]
                )

    def train(
        self,
        env,
        n_episodes=5000,
        show_progress=False,
        progress_desc="MC training",
        max_steps_per_episode=None,
    ):
        """Train for n_episodes. Returns list of total rewards per episode."""
        reward_history = []
        episode_iter = range(n_episodes)

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
            trajectory = self._collect_episode(env, max_steps_per_episode=max_steps_per_episode)
            self._update(trajectory)
            total_reward = sum(r for _, _, r in trajectory)
            reward_history.append(total_reward)

        return reward_history
