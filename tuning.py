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
        # "episodes" : [250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        "episodes": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        "alpha" : [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        "gamma" : [0.9, 0.95, 0.98, 0.99, 0.999],
        "epsilon" : [0.05, 0.1, 0.15, 0.2, 0.25]
    }

    q_learning = {
        "episodes" : [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        "alpha" : [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        "gamma" : [0.9, 0.95, 0.98, 0.99, 0.999],
        "epsilon" : [0.05, 0.1, 0.15, 0.2, 0.25]
    }

    mc_epsilon_greedy = {
        "episodes" : [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500],
        "gamma" : [0.9, 0.925, 0.95, 0.98],
        "epsilon" : [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    }


def evaluate(env, test_function, n, **kwargs):
    """
    runs the test function n times using the kwargs given.
    returns a dictionary of the arguments, and passrate
    """
    result = 0
    # runs the algorithm n times and averages the result to reduce noise/ randomness
    for _ in range(n):
        Q = test_function(env, **kwargs)
        policy = Q_to_policy(Q, env)
        result += test_policy(policy, env, silent=True)
    return {**kwargs, "pass_rate" : result / n} # this returns the parameters dict with pass_rate appended to it

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
    result = parallel_tuning(env, f, evaluate, parameters, n)
    return pd.DataFrame(result)

def plot_hyperparameters(df: pd.DataFrame, title = "undefined", show=False, save=True):
    hyperparams = set(df.columns) - {"pass_rate", "episodes"} # setminus to single out the hyperparameters, neat.

    _, axes = plt.subplots(1, len(hyperparams), figsize=(18, 5), sharey=True)

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

def extract_hyperparameters(df):
    """
    takes a dataframe of tuning testdata 
    returns a dict where the keys are the parameters and the values their optimal value
    """
    # filtered_df = df[df["pass_rate"] > 0.99]
    filtered_df = df[df["pass_rate"]==df["pass_rate"].max()]
    best_values = filtered_df[filtered_df["episodes"]==filtered_df["episodes"].min()].iloc[0].to_dict()
    best_values["episodes"] = int(best_values["episodes"]) # for some reasong episodes turns into a float here... 
    return best_values


if __name__ == "__main__":
    env = MDPGame()
    env.load_level(medium_level)

    df = test_hyperparameters(env, 1, TuningParameters.q_learning, q_learning)
    print(df)
    plot_hyperparameters(df, "q_learning")
