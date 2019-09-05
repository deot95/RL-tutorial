#############################################################################
# Copyright (C)                                                             #
# 2019 Daniel Ochoa - Just plot the epsilon policies (deot95@gmail.com)     #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                   #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                      #
# 2016 Artem Oboturov(oboturov@gmail.com)                                   #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                               #
# Permission given to modify the code as long as you keep this              #
# declaration at the top                                                    #
#############################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

class Bandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False,
                     true_reward=0.):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)
        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards

    
def plot_policy_performance(runs=2000, time=1000, epsilons = [0, 0.1, 0.01, 0.5], fromFile = None ):
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]
    if fromFile is None:
        best_action_counts, rewards = simulate(runs, time, bandits)
        np.savez('./bandit_simulation.npz', best_action_counts = best_action_counts, rewards=rewards)
    else:
        load_dict = np.load(fromFile)
        best_action_counts = load_dict['best_action_counts']
        rewards = load_dict['rewards']

    fig, axs = plt.subplots(2, 1, figsize=(10, 20))
    ax = axs[0]
    for eps, rewards in zip(epsilons, rewards):
        ax.plot(rewards, label='epsilon = %.02f' % (eps))
    ax.set_xlabel('steps')
    ax.set_ylabel('average reward')
    ax.legend()

    ax = axs[1]
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='epsilon = %.02f' % (eps))
    ax.set_xlabel('steps')
    ax.set_ylabel('% optimal action')
    ax.legend()
    plt.savefig('../imgs/epsilon_exloration.png')
    plt.close()

def main():
    plot_policy_performance()

main()