import monte_carlo
from game import MDPGame, hard_level, long_level
import numpy as np
import policy_iteration as pi
import monte_carlo as mc
from monte_carlo import Q_to_V
from monte_carlo_epsilon import mc_control_epsilon_greedy, q_to_policy
from q_learning import q_learning
from sarsa import sarsa
from utils import print_V, print_policy, test_policy, Q_to_policy

if __name__ == "__main__":
    env = MDPGame()
    # env.load_level(hard_level)
    # env.load_level(long_level)

    print("-----Welcome to Reinforcement Learning!-----")
    print("-----Select which algorithm you would like to inspect for our agent playing Surfing in the Subway:-----")
    option = input("Policy Iteration (1) \n"
                   "Value Iteration (2)\n"
                   "Monte Carlo Control Epsilon Greedy (3)\n"
                   "SARSA (4)\n"
                   "Q-Learning (5)\n"
                   "Stop playing (e) \n")

    if option == "1":
        optimal_policy, optimal_V = pi.policy_iteration(env)
    elif option == "2":
        optimal_policy, optimal_V = pi.value_iteration(env)
    elif option == "3":
        Q, optimal_policy = mc_control_epsilon_greedy(env, num_episodes=50000, epsilon=0.4, gamma=0.9)

        pi.simulate_agent(env, optimal_policy)
        optimal_V = Q_to_V(Q, env)
        print_V(optimal_V, env)

    elif option == "4":
        Q = sarsa(env, episodes = 50000, alpha = 0.1, gamma = 0.9, epsilon = 0.1)
        optimal_policy = Q_to_policy(Q, env)

        optimal_V = {state: np.max(list(action_values.values())) for state, action_values in Q.items()}

        print_policy(optimal_policy, env)
        success_rate = test_policy(optimal_policy, env, count_partial_success=True)
        print(f"SARSA success rate: {success_rate:.2f}")
    elif option == "5":
        Q = q_learning(env, episodes = 50000, alpha = 0.1, gamma = 0.9, epsilon = 0.1)
        optimal_policy = Q_to_policy(Q, env)

        optimal_V = {state: np.max(list(action_values.values())) for state, action_values in Q.items()}

        print_policy(optimal_policy, env)
        success_rate = test_policy(optimal_policy, env, count_partial_success=True)
        print(f"Q-learning success rate: {success_rate:.2f}")

    # Display the learned policy
    for state in sorted(optimal_policy.keys()):
        action = np.argmax(optimal_policy[state])  # Best action for each state
        print(f"State {state}: Best Action -> {env.actions[action]}")

    pi.inspect_policy(optimal_policy, env)
    pi.inspect_value_function(optimal_V)
    pi.print_policy_grid(optimal_policy, env)