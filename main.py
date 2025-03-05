from game import MDPGame
import numpy as np
import policy_iteration as pi

env = MDPGame()
optimal_policy, optimal_V = pi.policy_iteration(env)

pi.simulate_agent(env, optimal_policy)

# Display the learned policy
for state in sorted(optimal_policy.keys()):
    action = np.argmax(optimal_policy[state])  # Best action for each state
    print(f"State {state}: Best Action -> {env.actions[action]}")

# pi.plot_policy(env, optimal_policy)
# pi.inspect_policy(optimal_policy, env)
# pi.inspect_value_function(optimal_V)
pi.print_policy_grid(optimal_policy, env)