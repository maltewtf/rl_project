import numpy as np
import random
from collections import defaultdict


def mc_control_epsilon_greedy(env, num_episodes, epsilon, gamma):
    """
    Monte Carlo control with an ε-greedy strategy without exploring starts.

    This function uses first-visit Monte Carlo prediction to estimate action-values
    Q(s, a) and then improves the policy using an ε-greedy rule. Episodes start from
    the natural starting state provided by env.reset(), so no exploring starts are used.

    Parameters:
        env          : An instance of your MDPGame environment.
                       It must provide:
                           - reset(): returns an initial state.
                           - get_next_state(state, action): returns (next_state, reward, done).
                           - actions: a list of possible actions.
        num_episodes : Number of episodes for training.
        epsilon      : Exploration probability for the ε-greedy policy.
        gamma        : Discount factor.

    Returns:
        Q      : A dictionary mapping each state to a NumPy array of action-values.
                 For example, Q[s][a] is the estimated return from taking action a in state s.
        policy : A dictionary mapping each state to a one-hot encoded NumPy array that
                 represents the greedy action with respect to Q.
    """
    # Number of possible actions
    num_actions = len(env.actions)

    # Initialize Q(s,a) as a defaultdict of zero arrays of length num_actions.
    Q = defaultdict(lambda: np.zeros(num_actions))

    # For each state-action pair, track cumulative return and counts (for averaging)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for i_episode in range(num_episodes):
        episode = []
        # Start an episode with the environment's natural starting state
        state = env.reset(4)
        done = False

        # Generate an episode using the current ε-greedy policy.
        while not done:
            # Select an action according to the ε-greedy rule.
            if random.random() < epsilon:
                action_index = random.randint(0, num_actions - 1)
            else:
                action_index = int(np.argmax(Q[state]))
            action = env.actions[action_index]

            # Get the next state, reward, and terminal flag from the environment.
            next_state, reward, done = env.get_next_state(state, action)

            # Append this step to the episode (store state, action index, and reward).
            episode.append((state, action_index, reward))
            state = next_state

        # First-visit MC update: traverse the episode in reverse order.
        visited = set()  # to record (state, action_index) pairs already updated in this episode
        G = 0.0  # cumulative discounted return
        for (s, a, r) in reversed(episode):
            G = gamma * G + r
            # Update only on the first occurrence of (s, a)
            if (s, a) not in visited:
                returns_sum[(s, a)] += G
                returns_count[(s, a)] += 1.0
                Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]
                visited.add((s, a))

        # Optional: Print progress every 1000 episodes
        if (i_episode + 1) % 1000 == 0:
            print(f"Episode {i_episode + 1}/{num_episodes} completed.")

    # Derive the final (greedy) policy from Q:
    # For each state, choose the action with the highest Q-value.
    policy = {}
    for s, action_values in Q.items():
        best_action_index = int(np.argmax(action_values))
        one_hot = np.zeros(num_actions)
        one_hot[best_action_index] = 1.0
        policy[s] = one_hot

    return Q, policy

def q_to_policy(Q, env):
    """
    Converts a Q-function to a greedy policy.

    For each state, chooses the action with the maximum Q-value,
    and returns a dictionary mapping state -> action (as defined in env.actions).
    """
    policy = {}
    for state, action_values in Q.items():
        best_action_index = int(np.argmax(action_values))
        policy[state] = env.actions[best_action_index]
    return policy