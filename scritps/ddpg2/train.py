#!/usr/bin/env python3

import sys
import gym
import numpy as np
import pandas as pd
from PlutoEnv import Drone_state
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *

env = Drone_state()

agent = DDPGagent(env)
noise = OUNoise(env.action_space())
batch_size = 64
rewards = []
avg_rewards = []

for episode in range(5000):
	state = env.reset()
	rospy.sleep(1)
	noise.reset()
	episode_reward = 0
    	#while(True):
	for step in range(500):
		action = agent.get_action(state)
		action = noise.get_action(action, step)
		new_state, reward, done = env.step(action) 
		agent.memory.push(state, action, reward, new_state, done)
        
		if len(agent.memory) > batch_size:
			agent.update(batch_size)        
        
		state = new_state
		episode_reward += reward

		if done:
			sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
			break

	rewards.append(episode_reward)
	avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
