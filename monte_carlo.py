from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def mc_prediction(env, policy, gamma=0.99, num_episodes=500):
    V = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(num_episodes):
        episode = []
        state = env.reset()
        done = False

        while not done:
            action_probs = policy[state]
            action = np.random.choice(env.actions, p=action_probs)
            next_state, reward, done = env.get_next_state(state, action)
            episode.append((state, action, reward))
            state = next_state

        G = 0
        visited = set()
        for s, a, r in reversed(episode):
            G = r + gamma * G
            if s not in visited:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                visited.add(s)

    return V


def mc_control(env, gamma=0.99, epsilon=0.1, num_iterations=10, episodes_per_iteration=500):
    # Initialize a random policy
    policy = {}
    states, actions = env.get_states_actions()
    for state in states:
        policy[state] = np.ones(len(actions)) / len(actions)

    Q = defaultdict(lambda: np.zeros(len(actions)))
    returns = defaultdict(list)

    for iteration in range(num_iterations):
        V = mc_prediction(env, policy, gamma, episodes_per_iteration)

        # MC Control step (policy improvement)
        for state in states:
            Q_values = np.zeros(len(actions))

            for action_idx, action in enumerate(actions):
                next_state, reward, done = env.get_next_state(state, action)
                if done:
                    Q_values[action_idx] = reward
                else:
                    Q_values[action_idx] = reward + gamma * V[next_state]

            best_action = np.argmax(Q_values)

            # Epsilon-greedy policy update
            policy[state] = np.ones(len(actions)) * epsilon / len(actions)
            policy[state][best_action] += 1.0 - epsilon

            Q[state] = Q_values

        print(f"Iteration {iteration+1}/{num_iterations} completed.")

    return policy, Q_to_V(Q, env)


# Helper to convert Q to V
def Q_to_V(Q, env):
    V = {}
    for state in [(y, x) for y in range(env.height) for x in range(env.width)]:
        V[state] = max(Q[state]) if state in Q else 0

    return V