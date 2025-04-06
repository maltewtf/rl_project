import concurrent.futures
from utils import *
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from game import MDPGame, medium_level
from q_learning import q_learning
from utils import Q_to_policy


class TuningParameters:
    sarsa = {
        "episodes" : [500, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000],
        "alpha" : [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        "gamma" : [0.9, 0.95, 0.98, 0.99, 0.999],
        "epsilon" : [0.05, 0.1, 0.15, 0.2, 0.25]
    }

    q_learning = {
        "episodes" : [500, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000],
        "alpha" : [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        "gamma" : [0.9, 0.95, 0.98, 0.99, 0.999],
        "epsilon" : [0.05, 0.1, 0.15, 0.2, 0.25]
    }

    mc_epsilon_greedy = {
        "episodes" : [1000, 2000, 3000, 4000, 5000],
        "gamma" : [0.9, 0.95, 0.98, 0.99],
        "epsilon" : [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    }


def evaluate_TD(env, test_function, n, **kwargs):
    result = 0
    # runs the algorithm n times and averages the result to reduce noise/ randomness
    for _ in range(n):
        Q = test_function(env, **kwargs)
        policy = Q_to_policy(Q, env)
        result += test_policy(policy, env, silent=True)
    return {**kwargs, "pass_rate" : result / n} # this returns the parameters dict with pass_rate appended to it

def evaluate_DP(env, test_function, n, **kwargs):
    result = 0
    for _ in range(n):
        policy, _ = test_function(env, **kwargs)
        result += test_policy(policy, env, silent=True)
    return {**kwargs, "pass_rate": result / n}

def parallel_tuning(env, test_function, eval_function, parameters, n=1):
    results = []
    param_combinations = [dict(zip(parameters.keys(), values)) for values in product(*parameters.values())] # product will compute the powerset of the parameters (I know, its beautiful)
    
    # will launch a thread for each combination of parameters
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(eval_function, env, test_function, n, **params): params for params in param_combinations}
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results

def test_hyperparameters(env, n, parameters, f) -> pd.DataFrame:
    """makes call to parallel_tuning and packages it into a dataframe"""
    result = parallel_tuning(env, f, evaluate_TD, parameters, n)
    return pd.DataFrame(result)

def plot_hyperparameters(df: pd.DataFrame, title = "undefined", show=False, save=True):
    hyperparams = set(df.columns) - {"pass_rate", "episodes"} # setminus to single out the hyperparameters, neat.
    
    print(hyperparams)

    fig, axes = plt.subplots(1, len(hyperparams), figsize=(18, 5), sharey=True)

    sns.set_theme(font_scale=1.5)

    for i, hyperparameter in enumerate(hyperparams):
        # groups by the hyperparameters and only keeps the row with of that parameter which has the highes pass_rate 
        # (the idea is that this is the highest "potential" value that can be reached with this parametera)
        df_max = df.groupby([hyperparameter, 'episodes'], as_index=False)['pass_rate'].max()

        sns.lineplot(
            data=df_max,
            x='episodes',
            y='pass_rate',
            hue=hyperparameter,
            marker='o',
            palette='viridis',
            ax=axes[i]
        )

        axes[i].set_title(f'Pass Rate over Episodes by {hyperparameter.capitalize()}')
        axes[i].set_xlabel('Episodes', fontsize=20)
        axes[i].set_ylabel('Pass Rate' if i == 0 else '', fontsize=20)  # Only left plot shows y-axis label
        axes[i].legend(title=hyperparameter.capitalize())
        axes[i].grid(True)

    plt.tight_layout()

    if show:
        plt.show() # this is not necessary, but nice to have for the notebook
    
    if save:
        plt.savefig(f"graphs/{title}.png")

if __name__ == "__main__":
    env = MDPGame(random_x=True)
    env.load_level(medium_level)

    df = test_hyperparameters(env, 1, TuningParameters.q_learning, q_learning)
    print(df)
    plot_hyperparameters(df, "q_learning")
