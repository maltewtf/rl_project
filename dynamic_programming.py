import numpy as np
from game import MDPGame, hard_level
from utils import *

def policy_evaluation(env, policy, gamma=0.99, theta=1e-10):
    states, _ = env.get_states_actions()
    V = Types.V()

    delta = 1
    while delta > theta:
        delta = 0
        for state in states:
            v = V[state]
            action = policy[state]

            next_state, reward, done = env.step(state, action)

            if done:
                V[state] = reward
            else:
                V[state] = reward + gamma * V[next_state]

            delta = max(delta, abs(v - V[state]))

    return V

def policy_improvement(env, V, gamma=0.99):
    """Improves policy by choosing the greedy action based on V(s)."""
    policy = {}
    states, actions = env.get_states_actions()

    for state in states:
        Q_values = np.zeros(len(actions))  # Store Q-values for all actions

        for action_idx, action in enumerate(actions):
            next_state, reward, done = env.step(state, action)

            # Compute Q-values correctly
            Q_values[action_idx] = reward + gamma * V[next_state] * (not done)

        # Select the best action greedily
        best_action = np.argmax(Q_values)
        policy[state] = actions[best_action]

    return policy

def policy_iteration(env, gamma=0.99, theta=1e-6):
    """Finds the optimal policy using Policy Iteration."""

    # Initialize a random policy (equal probability for all actions)
    policy = Types.Policy()

    while True:
        V = policy_evaluation(env, policy, gamma, theta)  # Evaluate policy
        new_policy = policy_improvement(env, V, gamma)  # Improve policy

        # Check convergence
        if all(policy[s] == new_policy[s] for s in policy):
            break

        policy = new_policy  # Update policy

    return policy, V


def value_iteration(env, gamma=0.99, theta=1e-6):
    """Finds the optimal policy using Value Iteration."""
    V = Types.V()
    states, actions = env.get_states_actions()

    delta = 1
    while delta > theta:
        delta = 0

        for state in states:
            v = V[state]  # Expected return for this state

            q_values = np.zeros(len(actions))  # Store Q-values for all actions

            for action_idx, action in enumerate(actions):
                next_state, reward, done = env.step(state, action)

                # Compute Q-values correctly
                q_values[action_idx] = reward + gamma * V[next_state] * (not done)

            V[state] = max(q_values)  # Update state value

            delta = max(delta, abs(v - V[state]))

    # Extract the optimal policy
    policy = {}
    for state in states:
        q_values = np.zeros(len(actions))
        for action_idx, action in enumerate(actions):
            next_state, reward, done = env.step(state, action)
            q_values[action_idx] = reward + gamma * V[next_state] * (not done)
        best_action = np.argmax(q_values)
        policy[state] = actions[best_action]

    return policy, V

if __name__ == "__main__":
    env = MDPGame(random_x=True)
    env.load_level(hard_level)

    policy, V = policy_iteration(env, gamma=0.99)

    print_policy(policy, env)