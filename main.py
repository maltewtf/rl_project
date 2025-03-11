from game import MDPGame
import numpy as np
import policy_iteration as pi
import monte_carlo as mc

env = MDPGame()

print("-----Welcome to Reinforcement Learning!-----")
print("-----Select which algorithm you would like to inspect for our agent playing Surfing in the Subway:-----")
option = input("Policy Iteration (1) \n"
               "Monte Carlo Prediction (2) \n"
               "Monte Carlo Control (3)\n"
               "Stop playing (e) \n")

if option == "1":
    optimal_policy, optimal_V = pi.policy_iteration(env)
elif option == "2":
    V = mc.mc_prediction(env, num_sim=5000)
    optimal_policy, optimal_V = pi.policy_iteration(env)
    #optimal_policy = mc.mc_policy_improvement(env, V)
elif option == "3":
    Q = mc.mc_control(env, num_sim=5000)
    V = mc.Q_to_V(Q, env)
    optimal_policy, optimal_V = pi.policy_iteration(env)
    #optimal_policy = mc.mc_policy_improvement(env, V)

pi.simulate_agent(env, optimal_policy)

# Display the learned policy
for state in sorted(optimal_policy.keys()):
    action = np.argmax(optimal_policy[state])  # Best action for each state
    print(f"State {state}: Best Action -> {env.actions[action]}")

# pi.inspect_policy(optimal_policy, env)
# pi.inspect_value_function(optimal_V)
pi.print_policy_grid(optimal_policy, env)
