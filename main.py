from game import MDPGame, medium_level
from dynamic_programming import value_iteration, policy_iteration
from monte_carlo import mc_epsilon_greedy
from tuning import TuningParameters, test_hyperparameters, extract_hyperparameters, plot_hyperparameters
from q_learning import q_learning
from sarsa import sarsa
from utils import print_policy, test_policy, Q_to_policy
import sys

if __name__ == "__main__":
    args = sys.argv

    n = 5 # default n value
    env = MDPGame(random_x=True)
    env.load_level(medium_level)

    algorithms = ["value_iteration", "policy_iteration", "monte_carlo", "sarsa", "q_learning"]

    if "-n" in args:
        n = int(args[args.index("-n") + 1])
    if any([f"-{i}" in args for i in algorithms]):
        algorithms = [i[1:] for i in args if i[1:] in algorithms]

    print(algorithms)

    # Dynamic Programming
    if "value_iteraion" in algorithms:
        policy, V = value_iteration(env, gamma=0.95, theta=1e-10)
        print("policy generated by value iteration: ")
        test_policy(policy, env)
        print_policy(policy, env)

    if "policy_iteration" in algorithms:
        policy, V = policy_iteration(env, gamma=0.95, theta=1e-10)
        print("policy generated by policy iteration: ")
        test_policy(policy, env)
        print_policy(policy, env)

    # monte carlo
    if "monte_carlo" in algorithms:
        print(f"Testing monte carlo methods with {n} samples: ")
        df = test_hyperparameters(env, n, TuningParameters.mc_epsilon_greedy, mc_epsilon_greedy)
        plot_hyperparameters(df, "monte_carlo")
        ideal_params = extract_hyperparameters(df)
        print(f"ideal hyperparameters found {ideal_params}")
        print("policy generated by montecarlo methods, using ideal parameters: ")

        params = {**ideal_params, "env": env}
        params.pop("pass_rate") # not an input parameter
        Q = mc_epsilon_greedy(**params)
        policy = Q_to_policy(Q, env)
        test_policy(policy, env)
        print_policy(policy, env)


    # sarsa
    if "sarsa" in algorithms:
        print(f"Testing SARSA with {n} samples: ")
        df = test_hyperparameters(env, n, TuningParameters.sarsa, sarsa)
        plot_hyperparameters(df, "sarsa")
        ideal_params = extract_hyperparameters(df)
        print(f"ideal hyperparameters found {ideal_params}")
        print("policy generated by SARSA, using ideal parameters: ")

        params = {**ideal_params, "env": env}
        params.pop("pass_rate") # not an input parameter
        Q = sarsa(**params)
        policy = Q_to_policy(Q, env)
        test_policy(policy, env)
        print_policy(policy, env)

    # q-learning
    if "q_learning" in algorithms:
        print(f"Testing Q-learning with {n} samples: ")
        df = test_hyperparameters(env, n, TuningParameters.q_learning, q_learning)
        plot_hyperparameters(df, "q_learning")
        ideal_params = extract_hyperparameters(df)
        print(f"ideal hyperparameters found {ideal_params}")
        print("policy generated by SARSA, using ideal parameters: ")

        params = {**ideal_params, "env": env}
        params.pop("pass_rate") # not an input parameter
        Q = q_learning(**params)
        policy = Q_to_policy(Q, env)
        test_policy(policy, env)
        print_policy(policy, env)