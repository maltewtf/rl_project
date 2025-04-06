
# Reinforcement Learning project 2025

All code that that is needed for the project can be accessed by running main.py. Graphs will be stores in the graphs/ directory, the results of tuning as well as the generated policy will be printed into the console.

Testing is done with multiple samples to counteract randomness and reduce noise. It is recommended to run with a sample count of 10 or higher. Lower sample counts might result in tuning where an oulier is found but is not consitantly reproducable. Of course this comes at the cost of compute time, which is why the default value is set to 5. The same graphs and tuning can also be found in the notebook file, already pregenerated.

## Files

The project consits of multiple files. most imporantly dynamic_programming.py which contains all code neede for value and policy iteration. Then monte carlo which contains the code for monte carlo methods. sarsa.py and q_learning.py contain the algorithms for sarsa and q_learning while q_learning also contains some experimental TD functions.
The environment is described in game.py, with multiple different levels for testing. However we decided to only use "medium_level" to compare everything.
utils.py contains a lot of common useful functions, such as defining data types, printing policies, epsilon-greedy function or an argmax for dictionaries. 
tuning.py contains all the functions needed for hyperparameter tuning. We opted for a classical grid search which is very compute expensive, which is why it is parallelized and will easily make use of every CPU core it can get.

## Run Locally

To get all the needed modules run

```bash
  pip install -r requirements.txt
```

Simply running the ain.py will result in it testing and graphing all algorithms, with a default sample count of 5.

```bash
  python main.py
```

To change the amount of samples enter "-n" followed by a positive integer

```bash
  python main.py -n 20
```

specifying any of the algorithms will only run those 

```bash
  python main.py -n 20 -sarsa
```

these are all arguements that can be entered:

- "-n {int}"
- "-value_iteration"
- "-policy_iteration"
- "-monte_carlo"
- "-sarsa"
- "-q_learning"

