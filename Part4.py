import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import time

def initialize_environment(env_id):
    def _init():
        env = gym.make(env_id)
        return env
    return _init

cartpole_env = DummyVecEnv([initialize_environment('CartPole-v1')])
acrobot_env = DummyVecEnv([initialize_environment('Acrobot-v1')])

def train_rl_model(model_cls, env, steps=10000):
    model = model_cls('MlpPolicy', env, verbose=1)
    start = time.time()
    model.learn(total_timesteps=steps)
    duration = time.time() - start
    return model, duration

def evaluate_rl_model(model, env, episodes=100):
    rewards = []
    steps_to_converge = None
    total_steps = 0
    base_env = env.envs[0]
    for _ in range(episodes):
        observation = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0
        while not done:
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            episode_steps += 1
            total_steps += 1
            if steps_to_converge is None and reward >= base_env.spec.reward_threshold:
                steps_to_converge = total_steps
        rewards.append(total_reward)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    convergence_rate = steps_to_converge if steps_to_converge is not None else total_steps
    return mean_reward, std_reward, convergence_rate

train_steps = 10000

a2c_cartpole_model, a2c_cartpole_time = train_rl_model(A2C, cartpole_env, train_steps)
ppo_cartpole_model, ppo_cartpole_time = train_rl_model(PPO, cartpole_env, train_steps)
dqn_cartpole_model, dqn_cartpole_time = train_rl_model(DQN, cartpole_env, train_steps)

a2c_acrobot_model, a2c_acrobot_time = train_rl_model(A2C, acrobot_env, train_steps)
ppo_acrobot_model, ppo_acrobot_time = train_rl_model(PPO, acrobot_env, train_steps)
dqn_acrobot_model, dqn_acrobot_time = train_rl_model(DQN, acrobot_env, train_steps)

def print_evaluation(env, model, model_label, env_label):
    mean_reward, std_reward, convergence_rate = evaluate_rl_model(model, env)
    print(f"{env_label} Environment Evaluation ({model_label}):")
    print(f"Average reward: {mean_reward}, reward variance: {std_reward}, convergence rate: {convergence_rate}")

print_evaluation(cartpole_env, a2c_cartpole_model, "A2C", "CartPole")
print_evaluation(cartpole_env, ppo_cartpole_model, "PPO", "CartPole")
print_evaluation(cartpole_env, dqn_cartpole_model, "DQN", "CartPole")

print_evaluation(acrobot_env, a2c_acrobot_model, "A2C", "Acrobot")
print_evaluation(acrobot_env, ppo_acrobot_model, "PPO", "Acrobot")
print_evaluation(acrobot_env, dqn_acrobot_model, "DQN", "Acrobot")
