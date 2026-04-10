"""env_utils.py - helpers to create and interact with TextFlappyBird-v0."""

import gymnasium as gym
import text_flappy_bird_gym  # noqa: registers envs

DEFAULT_MAX_EPISODE_STEPS = 2_000


def make_env(height=15, width=20, pipe_gap=4, max_episode_steps=DEFAULT_MAX_EPISODE_STEPS):
    """Create a TextFlappyBird-v0 environment with a hard episode cap."""
    env = gym.make(
        "TextFlappyBird-v0",
        height=height,
        width=width,
        pipe_gap=pipe_gap,
    )
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def get_state(obs):
    """Convert raw observation to a hashable state tuple.

    TextFlappyBird-v0 returns (dx, dy) - integer distances to the
    nearest pipe gap center. Already discrete, so just cast to tuple.
    """
    return tuple(int(x) for x in obs)
