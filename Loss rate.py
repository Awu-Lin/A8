import gym
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.001
discount_rate = 0.99
exploration_rate = 0.1
min_exploration_rate = 0.01
exploration_decay_rate = 0.995
num_episodes = 200
max_timesteps = 1000

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
    return tuple(np.digitize(val, b) - 1 for val, b in zip(state, bins))

def q_learning_algorithm(env, episodes, timesteps, alpha, gamma, epsilon, epsilon_min, epsilon_decay, bins):
    q_values = np.zeros([len(bins[0]), len(bins[1]), len(bins[2]), len(bins[3]), len(bins[4]), len(bins[5]), len(bins[6]), len(bins[7]), env.action_space.n])
    average_losses = []

    for episode in range(episodes):
        state = env.reset()
        state = discretize_state(state, bins)
        episode_loss = []
        
        for step in range(timesteps):
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(q_values[state])

            next_state, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, bins)
            
            best_next_action = np.argmax(q_values[next_state])
            td_error = reward + gamma * q_values[next_state][best_next_action] - q_values[state][action]
            q_values[state][action] += alpha * td_error
            episode_loss.append(abs(td_error))
            
            state = next_state
            
            if done or truncated:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        average_losses.append(np.mean(episode_loss))
    
    return average_losses

def sarsa_algorithm(env, episodes, timesteps, alpha, gamma, epsilon, epsilon_min, epsilon_decay, bins):
    q_values = np.zeros([len(bins[0]), len(bins[1]), len(bins[2]), len(bins[3]), len(bins[4]), len(bins[5]), len(bins[6]), len(bins[7]), env.action_space.n])
    average_losses = []

    for episode in range(episodes):
        state = env.reset()
        state = discretize_state(state, bins)
        episode_loss = []
        
        if np.random.rand() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = np.argmax(q_values[state])
        
        for step in range(timesteps):
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, bins)
            
            if np.random.rand() < epsilon:
                next_action = np.random.choice(env.action_space.n)
            else:
                next_action = np.argmax(q_values[next_state])

            td_error = reward + gamma * q_values[next_state][next_action] - q_values[state][action]
            q_values[state][action] += alpha * td_error
            episode_loss.append(abs(td_error))
            
            state, action = next_state, next_action
            
            if done or truncated:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        average_losses.append(np.mean(episode_loss))
    
    return average_losses

environment = gym.make("LunarLander-v2", new_step_api=True)

q_learning_avg_losses = q_learning_algorithm(environment, num_episodes, max_timesteps, learning_rate, discount_rate, exploration_rate, min_exploration_rate, exploration_decay_rate, state_bins)

sarsa_avg_losses = sarsa_algorithm(environment, num_episodes, max_timesteps, learning_rate, discount_rate, exploration_rate, min_exploration_rate, exploration_decay_rate, state_bins)

plt.plot(range(1, num_episodes + 1), q_learning_avg_losses, label='Q-learning')
plt.plot(range(1, num_episodes + 1), sarsa_avg_losses, label='SARSA')
plt.xlabel('Episode')
plt.ylabel('Average Loss')
plt.title('Q-learning vs SARSA on Lunar Lander (Loss)')
plt.legend()
plt.show()
