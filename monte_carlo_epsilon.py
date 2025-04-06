import numpy as np
from collections import defaultdict
from game import *
from utils import *

def mc_epsilon_greedy(env, episodes, epsilon, gamma):
    """
    Monte Carlo control with an ε-greedy strategy without exploring starts.

    This function uses first-visit Monte Carlo prediction to estimate action-values
    Q(s, a) and then improves the policy using an ε-greedy rule. Episodes start from
    the natural starting state provided by env.reset(), so no exploring starts are used.

    Parameters:
        env          : An instance of your MDPGame environment.
                       It must provide:
                           - reset(): returns an initial state.
                           - step(state, action): returns (next_state, reward, done).
                           - actions: a list of possible actions.
        num_episodes : Number of episodes for training.
        epsilon      : Exploration probability for the ε-greedy policy.
        gamma        : Discount factor.

    Returns:
        Q      : A dictionary mapping each state to a NumPy array of action-values.
                 For example, Q[s][a] is the estimated return from taking action a in state s.
    """

    Q = Types.Q()

    returns = defaultdict(list)

    for i_episode in range(episodes):
        episode = []
        # Start an episode in a random location to ensure maximum exploarabiliy
        state = env.reset()
        done = False

        # Generate an episode using the current ε-greedy policy.
        while not done:
            # generat A_t and use it to get S_t+1, R_t+1
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = env.step(state, action)

            episode.append((state, action, reward))
            state = next_state

        # First-visit MC update: traverse the episode in reverse order.
        visited = set()  # to record (state, action_index) pairs already updated in this episode
        G = 0  # cumulative discounted return
        for state, action, reward in reversed(episode):
            G = gamma * G + reward

            # Update only on the first occurrence of (s, a)
            if not (state, action) in visited:
                returns[(state, action)].append(G)
                Q[state][action] = np.average(returns[(state, action)])

                visited.add((state, action))

    return Q
        

if __name__ == "__main__":
    env = MDPGame(random_x=True)
    env.load_level(hard_level)
    Q = mc_epsilon_greedy(env, 2000, epsilon=0.2, gamma=0.99)

    test_policy(Q_to_policy(Q, env), env)
    print_policy(Q_to_policy(Q, env), env)

