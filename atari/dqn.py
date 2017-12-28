#pylint: disable-msg=C0103, C0411, C0413
import numpy as np
import pprint
import sys
if '../' not in sys.path:
    sys.path.append('../')
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment
            env.nA is a number of actions in the environment
        theta: we stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    while True:
        delta = 0
        for s in range(env.nS):
            action_value = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_value[a] += prob * (
                        reward + discount_factor * V[next_state])
            best_action = np.argmax(action_value)
            policy[s] = np.eye(env.nA)[best_action]
            value = np.max(action_value)
            delta = np.max((delta, abs(value - V[s])))
            V[s] = value
        if delta < theta:
            return policy, V


policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")
