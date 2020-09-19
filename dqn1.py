import gym
import numpy as np
from nn import NeuralNet as net

env = gym.make('CartPole-v0')
state = env.reset()

model1 = net(env.reset().shape[0], 20, env.action_space.n, 0.3) # Q (action - value function)
model2 = net(model1.inodes, model1.hnodes, model1.onodes, model1.lr) # Q cap (target action-value function)
model2.wih = model1.wih
model2.who = model2.who


print(model1.inodes, model1.onodes)

print(model1.query(env.reset()))
outputs = np.zeros(env.action_space.n)

outputs[env.action_space.sample()] = 1.0
for i in range(1000):
	action = env.action_space.sample()
	outputs[action] = 1
	model1.train(state, action)
	nextState, reward, done, _ = env.step(action)
	state = nextState
	if done:
		env.reset()
print(model1.query(env.reset()))

# for i in range(100):
# 	env.render()
# 	s = env.step(env.action_space.sample())
# 	print(f"Next State {s[0]}")
# 	if s[2]:
# 		env.reset()
# env.close()