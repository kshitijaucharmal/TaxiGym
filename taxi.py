import numpy as np 
import os
from time import sleep
import gym

env = gym.make('Taxi-v3')
env.reset()

n_sample_games = 10
n_episodes = 8000

render = True # for training

Q = np.zeros([env.observation_space.n, env.action_space.n])

epsilon = 0.6
alpha = 0.6
gamma = 0.99

# training
for episode in range(n_episodes):
    state = env.reset()
    done = False
    rewards = 0
    while not done:
        action = np.argmax(Q[state]) if np.random.random() < epsilon else env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        Q[state, action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state, action])
        rewards += reward
        state = next_state
    if episode % 500 == 0:
        print('Episode {} yielded {} rewards'.format(episode, rewards))

for i in range(n_sample_games):
    state = env.reset()
    done = False
    rewards = 0
    while not done:
        if render:
            env.render()
        print(f"Episode {i}: {rewards}")
        sleep(0.1)
        os.system('clear') # for windows replace clear with cls
        action = np.argmax(Q[state])
        next_state, reward, done, _ = env.step(action)
        state = next_state
        rewards += reward

env.close()