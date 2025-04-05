import concurrent.futures
from utils import *
import concurrent.futures
from itertools import product

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