import numpy as np
import matplotlib.pyplot as plt


def get_gambler_env():
    """
    Get the environment of gambler
    Returns:
        P: P[s][a]
    """
    P = []
    for s in range(99):
        P.append([])
        state = s + 1
        action_range = np.min((state, 100 - state))
        for a in range(action_range + 1):
            P[s].append([])
            next_state = s + a
            if next_state not in range(99):
                P[s][a].append((0.4, next_state, 1, True))
            else:
                P[s][a].append((0.4, next_state, 0, False))
            next_state = s - a
            if next_state not in range(99):
                P[s][a].append((0.6, next_state, 0, True))
            else:
                P[s][a].append((0.6, next_state, 0, False))

    return P


def value_iter(P, discount_factor=1.0, theta=0.00001):
    """
    Value iteration, find the optimal value function of the optimal Policy
    """
    policy = random_policy()
    V = np.zeros(99)
    while True:
        delta = 0
        for s in range(99):
            state = s + 1
            action_range = np.min((state, 100 - state))
            value = np.zeros(action_range + 1)
            ##choose best action
            for a in range(action_range + 1):
                for (prob, next_state, reward, done) in P[s][a]:
                    print(prob, next_state, reward, done, a)
                    if done:
                        value[a] += prob * reward
                        continue
                    value[a] += prob * (
                        reward + discount_factor * V[next_state])
            best_action = np.argmax(value)
            print(s, best_action)
            best_value = np.max(value)
            print(value)
            policy[s] = np.eye(51)[best_action]
            delta = np.max((delta, abs(V[s] - best_value)))
            V[s] = best_value
            print(V)

        if delta < theta:
            return policy, V


def random_policy():
    policy = np.random.random([99, 51])
    for s in range(99):
        state = s + 1
        action_range = np.min((state, 100 - state))
        for a in range(51):
            if a > (action_range + 1):
                policy[s][a] = 0
    policy = (policy.T / np.mean(policy.T, 0)).T
    return policy


P = get_gambler_env()
policy, value = value_iter(P)
action = np.argmax(policy, 1)
print(action)
plt.plot(value)
plt.plot(action)
plt.show()
