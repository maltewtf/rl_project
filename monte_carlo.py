from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def mc_prediction(env, gamma=0.99, num_sim=500):
    """Monte Carlo prediction algorithm that estimates state values."""
    V = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(num_sim):
        episode = env.sample_episode()
        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G

            if s not in visited:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                visited.add(s)

    return V

def mc_control(env, gamma=0.99, epsilon=0.1, num_sim=500):
    """Monte Carlo Control"""
    Q = defaultdict(lambda: np.zeros(len(env.actions)))
    returns = defaultdict(list)

    for _ in range(num_sim):
        episode = env.sample_episode()
        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G

            if (s, a) not in visited:
                returns[(s, a)].append(G)
                Q[s][env.actions.index(a)] = np.mean(returns[(s, a)])
                visited.add((s, a))

        for s in Q:
            if np.random.rand() < epsilon:
                best_action = np.random.choice(env.actions)
            else:
                best_action = env.actions[np.argmax(Q[s])]

            Q[s] = np.eye(len(env.actions))[env.actions.index(best_action)]

    return Q

def mc_policy_improvement(env, V, gamma=0.99):
    """Derives a greedy policy from the estimated value function V"""
    policy = {}
    states, actions = env.get_states_actions()

    for state in states:
        Q_values = np.zeros(len(actions))

        for action_idx, action in enumerate(actions):
            next_state, reward, done = env.get_next_state(state, action)

            if env.level[next_state[0]][next_state[1]] == 1:
                Q_values[action_idx] = -float('inf')
            else:
                Q_values[action_idx] = reward + gamma * V[next_state] * (not done)

        best_action = np.argmax(Q_values)
        policy[state] = np.eye(len(actions))[best_action]

    return policy

def Q_to_V(Q, env):
    """Convert Q(s, a) to V(s) by taking max Q-value for each state."""
    V = {}
    for state in [(y, x) for y in range(env.height) for x in range(env.width)]:
        V[state] = max(Q[state]) if state in Q else 0

    return V