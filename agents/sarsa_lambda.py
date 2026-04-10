"""sarsa_lambda.py - Sarsa(lambda) with accumulating eligibility traces.

Implements the algorithm from Sutton & Barto, Section 12.7
(Sarsa(lambda) with eligibility traces, tabular version).
"""

import random
from collections import defaultdict

from env_utils import get_state

_TRACE_THRESHOLD = 1e-6  # prune traces below this to keep E small


class SarsaLambdaAgent:
    """Tabular Sarsa(lambda) agent with eps-greedy policy."""

    def __init__(self, n_actions=2, epsilon=0.1, alpha=0.1, gamma=0.99, lambda_=0.8):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_

        self.Q = defaultdict(lambda: [0.0] * n_actions)
        # Eligibility traces stored as {(state, action): float} - reset each episode
        self.E = {}

    def select_action(self, state):
        """eps-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(max(range(self.n_actions), key=lambda a: self.Q[state][a]))

    def train(
        self,
        env,
        n_episodes=5000,
        show_progress=False,
        progress_desc="Sarsa(lambda) training",
        max_steps_per_episode=None,
    ):
        """Train for n_episodes using online Sarsa(lambda) updates.

        Returns list of total rewards per episode.
        """
        reward_history = []
        decay = self.gamma * self.lambda_
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
            self.E = {}   # reset traces each episode
            obs, _ = env.reset()
            state = get_state(obs)
            action = self.select_action(state)

            total_reward = 0.0
            done = False
            steps = 0

            while not done:
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
                if step_limit is not None and steps >= step_limit:
                    done = True
                next_state = get_state(obs)
                next_action = None if done else self.select_action(next_state)

                # TD error
                delta = (
                    reward
                    + (0.0 if done else self.gamma * self.Q[next_state][next_action])
                    - self.Q[state][action]
                )

                # Accumulate trace for current (s, a)
                key = (state, action)
                self.E[key] = self.E.get(key, 0.0) + 1.0

                # Update Q and decay all traces; prune negligible ones
                alpha_delta = self.alpha * delta
                to_delete = []
                for k, e in self.E.items():
                    s, a = k
                    self.Q[s][a] += alpha_delta * e
                    new_e = e * decay
                    if new_e < _TRACE_THRESHOLD:
                        to_delete.append(k)
                    else:
                        self.E[k] = new_e
                for k in to_delete:
                    del self.E[k]

                total_reward += reward
                state = next_state
                action = next_action

            reward_history.append(total_reward)

        return reward_history
