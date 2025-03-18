from game import MDPGame, hard_level, long_level
import numpy as np
import policy_iteration as pi
import monte_carlo as mc
from monte_carlo import Q_to_V
from monte_carlo_epsilon import mc_control_epsilon_greedy, q_to_policy
from utils import print_V


if __name__ == "__main__":
    env = MDPGame()
    # env.load_level(hard_level)
    # env.load_level(long_level)

    print("-----Welcome to Reinforcement Learning!-----")
    print("-----Select which algorithm you would like to inspect for our agent playing Surfing in the Subway:-----")
    option = input("Policy Iteration (1) \n"
                   # "Monte Carlo Prediction (2) \n"
                   # "Monte Carlo Control (3)\n"
                   "Value Iteration (4)\n"
                   "Monte Carlo Control Epsilon Greedy (5)\n"
                   "Stop playing (e) \n")

    if option == "1":
        optimal_policy, optimal_V = pi.policy_iteration(env)
    elif option == "4":
        optimal_policy, optimal_V = pi.value_iteration(env)
    elif option == "5":
        Q, optimal_policy = mc_control_epsilon_greedy(env, num_episodes=400000, epsilon=0.4, gamma=0.9)

        pi.simulate_agent(env, optimal_policy)
        optimal_V = q_to_policy(Q, env)
        print_V(optimal_V, env)

    # Display the learned policy
    for state in sorted(optimal_policy.keys()):
        action = np.argmax(optimal_policy[state])  # Best action for each state
        print(f"State {state}: Best Action -> {env.actions[action]}")

    pi.inspect_policy(optimal_policy, env)
    pi.inspect_value_function(optimal_V)
    pi.print_policy_grid(optimal_policy, env)