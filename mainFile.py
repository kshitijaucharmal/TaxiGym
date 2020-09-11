import gym
import numpy as np
import time

def loadFile():
	global Q
	K = []
	for i in range(env.observation_space.n):
		K.append([0.0 for j in range(env.action_space.n)])
	with open("file.txt", 'r') as f:
		d = f.readlines()
		s = []
		for i in range(len(d)):
			s.append(float(d[i]))
		s = np.array(s)
		for i in range(len(Q)):
			for j in range(len(Q[0])):
				Q[i, j] = s[i * len(Q[0]) + j]

def saveFile():
	with open("file.txt", 'w') as f:
		for i in range(len(Q)):
			for j in range(len(Q[i])):
				f.write(str(Q[i, j]) + "\n")

env = gym.make("FrozenLake-v0")
# parameters
alpha = 0.6
gamma = 0.9
epsilon = 0.6
max_episodes = 10000
train = False
Q = np.zeros([env.observation_space.n, env.action_space.n])

action = 0

if(train):
    for episode in range(max_episodes):
        done = False
        state = env.reset()
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            nextState, reward, done, _ = env.step(action)
            Q[state, action] += alpha * (reward + gamma * max(Q[nextState]) - Q[state, action])
            state = nextState
        print(f"Episode {episode}")
    saveFile()

else:
    loadFile()
    for i in range(10):
        done = False
        state = env.reset()
        while not done:
            action = np.argmax(Q[state])
            nextState, reward, done, _ = env.step(action)
            state = nextState
        print(f"Training Episode {i}")

env.close()