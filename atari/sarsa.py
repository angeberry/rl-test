import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if '../' not in sys.path:
    sys.path.append('../')

from collections import defaultdict
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = WindyGridworldEnv()


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy Policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to fun for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float between 0 and 1.

    Returns:
        A tuple(Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
