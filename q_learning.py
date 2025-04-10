from game import *
from utils import *

def q_learning(env: MDPGame, episodes, alpha, gamma, epsilon):
    """
    Applies the q_learning algorthim to a given environment and returns the Q-values in form of a dictionary.

    Parameters:
        env: MDPGame
        episodes: int
        alpha: float
        gamma: float
        epsilon: float

    returns:
        Q: defaultdict(lambda: defaultdict(int))
    """

    Q = Types.Q() # Q(s, a), more like Q = s -> a -> q-value

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = env.step(state, action)
            best_action = argmax(Q[next_state])

            # Q(S, A) += alpha * (R + gamma * Q(S', a) - Q(S, A))
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_action] - Q[state][action])

            state = next_state

    return Q

def double_q_learning(env: MDPGame, episodes, alpha, gamma, epsilon):
    """
    applies that double q-learning approach. This is technically not part of the project but was in the lectures.
    Its supposed to converge faster but from testing it about performes the same in this environment.

    Parameters:
        env: MDPGame
        episodes: int
        alpha: float
        gamma: float
        epsilon: float

    returns:
        Q: defaultdict(lambda: defaultdict(int))
    """

    Q1 = Types.Q() # Q(s, a)
    Q2 = Types.Q()

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = n_epsilon_greedy_policy([Q1, Q2], state, epsilon) # this is necessary because summing Q1 and Q2 every interation results in a massive slow down
            # action = epsilon_greedy_policy(sum_Q([Q1, Q2]), state, epsilon)

            next_state, reward, done = env.step(state, action)
            if random.uniform(0, 1) < .5:
                best_action = argmax(Q1[next_state])
                Q1[state][action] += alpha * (reward + gamma * Q1[next_state][best_action] - Q1[state][action])
            else:
                best_action = argmax(Q2[next_state])
                Q2[state][action] += alpha * (reward + gamma * Q2[next_state][best_action] - Q2[state][action])

            state = next_state

    return sum_Q([Q1, Q2])

def q_learning_until_pass(env: MDPGame, expected_pass_rate, alpha, gamma, epsilon):
    """
    runs q_learning episodes until it manages to pass with a 100% passrate. useful for testing and tuning.

    Parameters:
        env: MDPGame
        expected_pass_rate: float
        alpha: float
        gamma: float
        epsilon: float

    returns:
        Q: defaultdict(lambda: defaultdict(int))
    """
    assert expected_pass_rate < 1
    pass_rate = 0
    episodes = 0
    Q = Types.Q() # Q(s, a)

    while pass_rate < expected_pass_rate:
        episodes += 1
        state = env.reset()
        done = False

        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = env.step(state, action)
            best_action = argmax(Q[next_state])

            # Q(S, A) += alpha * (R + gamma * Q(S', a) - Q(S, A))
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_action] - Q[state][action])

            state = next_state

        pass_rate = test_policy(Q_to_policy(Q, env), env, silent=True, count_partial_success=True)

    print(f"episodes: {episodes}")
    return Q

if __name__ == "__main__":
    env = MDPGame(random_x=True)
    env.load_level(hard_level)

    Q = q_learning_until_pass(env, expected_pass_rate=0.99, alpha=0.1, gamma=0.99, epsilon=0.1)
    policy = Q_to_policy(Q, env)
    print_policy(policy, env)