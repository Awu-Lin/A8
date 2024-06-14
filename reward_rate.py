import gym
import numpy as np
import matplotlib.pyplot as plt

environment = gym.make("LunarLander-v2", new_step_api=True)

learning_rate = 0.001
discount_factor = 0.99
exploration_rate = 0.1
min_exploration_rate = 0.01
exploration_decay_rate = 0.995
num_episodes = 200
max_steps = 1000

state_bins = [
    np.linspace(-1.5, 1.5, 10),
    np.linspace(-1.5, 1.5, 10),
    np.linspace(-5, 5, 10),
    np.linspace(-5, 5, 10),
    np.linspace(-np.pi, np.pi, 10),
    np.linspace(-5, 5, 10),
    np.array([0, 1]),
    np.array([0, 1])
]

def discretize_state(state, bins):
    return tuple(np.digitize(s, b) - 1 for s, b in zip(state, bins))

def q_learning_algo(env, episodes, steps, alpha, gamma, epsilon, epsilon_min, epsilon_decay, bins):
    q_vals = np.zeros([len(bins[0]), len(bins[1]), len(bins[2]), len(bins[3]), len(bins[4]), len(bins[5]), len(bins[6]), len(bins[7]), env.action_space.n])
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        state = discretize_state(state, bins)
        episode_reward = 0
        
        for step in range(steps):
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(q_vals[state])

            next_state, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, bins)
            episode_reward += reward
            
            best_next_action = np.argmax(q_vals[next_state])
            td_error = reward + gamma * q_vals[next_state][best_next_action] - q_vals[state][action]
            q_vals[state][action] += alpha * td_error
            
            state = next_state
            
            if done or truncated:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(episode_reward)
    
    return rewards

def sarsa_algo(env, episodes, steps, alpha, gamma, epsilon, epsilon_min, epsilon_decay, bins):
    q_vals = np.zeros([len(bins[0]), len(bins[1]), len(bins[2]), len(bins[3]), len(bins[4]), len(bins[5]), len(bins[6]), len(bins[7]), env.action_space.n])
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        state = discretize_state(state, bins)
        episode_reward = 0
        
        if np.random.rand() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = np.argmax(q_vals[state])
        
        for step in range(steps):
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, bins)
            episode_reward += reward
            
            if np.random.rand() < epsilon:
                next_action = np.random.choice(env.action_space.n)
            else:
                next_action = np.argmax(q_vals[next_state])

            td_error = reward + gamma * q_vals[next_state][next_action] - q_vals[state][action]
            q_vals[state][action] += alpha * td_error
            
            state, action = next_state, next_action
            
            if done or truncated:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(episode_reward)
    
    return rewards

q_learning_rewards = q_learning_algo(environment, num_episodes, max_steps, learning_rate, discount_factor, exploration_rate, min_exploration_rate, exploration_decay_rate, state_bins)
sarsa_rewards = sarsa_algo
