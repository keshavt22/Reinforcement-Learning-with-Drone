#!/usr/bin/env python3

import sys
import gym
import pylab
import random
import numpy as np
import tensorflow
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

import math
import copy
import csv

#Ros packages
from plutodrone.msg import *
#from pid_tune.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int32
import rospy
import time

EPISODES = 3000

class Drone_state():
	def __init__(self):
		self.act_spc=[]
		for i in range(0,41):
			self.act_spc.append(i)
		random.shuffle(self.act_spc)	
		self.done = False
		rospy.init_node('pluto_fly', disable_signals = True)
		rospy.Subscriber('whycon/poses', PoseArray, self.cb_drone_state)
		self.pluto_cmd = rospy.Publisher('/drone_command', PlutoMsg, queue_size=10)
		self.cmd = PlutoMsg()
		self.cmd.rcRoll = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcThrottle = 1500
		self.cmd.rcAUX1 = 1500
		self.cmd.rcAUX2 = 1500
		self.cmd.rcAUX3 = 1500
		self.cmd.rcAUX4 = 1000
		self.cmd.plutoIndex = 0	
		self.z_hold = 30.12
		self.prev_z = 30.12
		self.x = 0.0
		self.y = 0.0
		self.drone_state = [0.0, 0.0]
			
	def action_space(self):
		return np.array(self.act_spc)
	
	def observation_space(self):
		return np.array(self.drone_state)	
		
	def reset(self):
		self.cmd.rcAUX1=1800
		self.pluto_cmd.publish(self.cmd)
		#rospy.sleep(1)
		self.cmd.rcAUX1=1500	
		return self.get_drone_state()		
	
	def throttle(self, t):
		self.cmd.rcThrottle = t
		self.pluto_cmd.publish(self.cmd)	
		
	def step(self, action):
		action = 1300 + (action*10)
		#print('action:', action)	
		self.done = False
		current_state = self.get_drone_state()
		#print("Current state : ", current_state)
		self.throttle(action)
		rospy.sleep(0.04)						
		next_state = self.get_drone_state()
		#print("Next state : ", next_state)
		self.prev_z = next_state[0]
		reward = self.get_reward(next_state)
		if((next_state[0]<22 or next_state[0]>38) or abs(self.x)>7 or abs(self.y)>6):
			self.done = True

		return next_state, reward, self.done
		
	def get_reward(self, state):
		reward = 0.0
		if ((abs(state[0] - self.z_hold) < 1)) :
			# goal reached 
			reward = 200
		elif (abs(state[0] - self.z_hold) < 2) :
			reward = 50
		elif (abs(state[0] - self.z_hold) < 3) :
			reward = 20
		elif (abs(state[0] - self.z_hold) < 4) :
			reward = 10		
		elif(state[0]<22 or state[0]>38):
			reward = -100										
		else:
			reward = -50
	# if (state[0] < -0.05 or state[0] > 5.05 or state[1] < -0.05 or state[1] > 5.05):
	
		if(state[1] < 12.5):
			reward = reward + 20
		elif(abs(state[1]) >= 12.5 and abs(state[1]) < 25.0):
			reward = reward + 5
		elif(abs(state[1]) >= 25.0 and abs(state[1]) < 37.5):
			reward = reward - 5
		elif(abs(state[1]) >= 75):
			reward = reward - 50
		else:
			reward = reward - 10				
		return reward		
		
	def set_drone_state(self, state):
		self.drone_state = state

	def get_drone_state(self):
		return self.drone_state	
		
	def cb_drone_state(self,pose):
		self.x = pose.poses[0].position.x
		self.y = pose.poses[0].position.y
		n=pose.poses[0].position.z
		nv = (self.prev_z - n)/0.04
		self.set_drone_state([n, nv])   #set z value obtained in bracket


# Double DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
	def __init__(self, state_size, action_size):
		self.load_model = False
		# get size of state and action
		self.state_size = state_size
		self.action_size = action_size

		# these is hyper parameters for the Double DQN
		self.discount_factor = 0.99
		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_decay = 0.999
		self.epsilon_min = 0.01
		self.epsilon_max = 1.0
		self.batch_size = 64
		self.train_start = 1000
		self.episode = 0
		# create replay memory using deque
		self.memory = deque(maxlen=100000)

		# create main model and target model
		self.model = self.build_model()
		self.target_model = self.build_model()

		# initialize target model
		self.update_target_model()

		if self.load_model:
			self.model.load_weights("drone_ddqn.h5")

	# approximate Q function using Neural Network
	# state is input and Q Value of each action is output of network
	def build_model(self):
		model = Sequential()
		model.add(Dense(150, input_dim=self.state_size, activation='tanh', kernel_initializer='he_uniform'))
		model.add(Dense(300, activation='tanh', kernel_initializer='he_uniform'))
		model.add(Dense(400, activation='selu', kernel_initializer='he_uniform'))
		model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
		model.summary()
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	# after some time interval update the target model to be same with model
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	# get action from model using epsilon-greedy policy
	def get_action(self, state, ep):
		self.episode = ep	
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		else:
			q_value = self.model.predict(state)
			return np.argmax(q_value[0])

	# save sample <s,a,r,s'> to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))
		#if self.epsilon > self.epsilon_min:
			#self.epsilon *= self.epsilon_decay
		if (self.episode < 800):
			self.epsilon = 1
		else :
			self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min)*(math.exp(-0.01*(self.episode-500)))	

	# pick samples randomly from replay memory (with batch_size)
	def train_model(self):
		if len(self.memory) < self.train_start:
			return
		batch_size = min(self.batch_size, len(self.memory))
		mini_batch = random.sample(self.memory, batch_size)
		update_input = np.zeros((batch_size, self.state_size))
		update_target = np.zeros((batch_size, self.state_size))
		action, reward, done = [], [], []

		for i in range(batch_size):
			update_input[i] = mini_batch[i][0]
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])
			update_target[i] = mini_batch[i][3]
			done.append(mini_batch[i][4])

		target = self.model.predict(update_input)
		target_next = self.model.predict(update_target)
		target_val = self.target_model.predict(update_target)

		for i in range(self.batch_size):
			# like Q Learning, get maximum Q value at s'
			# But from target model
			if done[i]:
				target[i][action[i]] = reward[i]
			else:
				# the key point of Double DQN
				# selection of action is from model
				# update is from target model
				a = np.argmax(target_next[i])
				target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][a])

		# make minibatch which includes target q value and predicted q value
		# and do the model fit!
		self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)


if __name__ == "__main__":
	# In case of CartPole-v1, you can play until 500 time step
	env = Drone_state()
	# get size of state and action from environment
	state_size = env.observation_space().shape[0]
	action_size = env.action_space().shape[0]

	agent = DoubleDQNAgent(state_size, action_size)

	scores, episodes = [], []

	for e in range(EPISODES):
		done = False
		score = 0
		state = env.reset()
		rospy.sleep(1)
		state = np.reshape(state, [1, state_size])

		while not done:

			# get action for the current state and go one step in environment
			action = agent.get_action(state, e)
			next_state, reward, done = env.step(action)
			next_state = np.reshape(next_state, [1, state_size])
			# if an action make the episode end, then gives penalty of -100
			#reward = reward if not done or score == 499 else -100
			if done:
				reward = -100
			# save the sample <s, a, r, s'> to the replay memory
			agent.append_sample(state, action, reward, next_state, done)
			# every time step do the training
			agent.train_model()
			score += reward
			state = next_state

			if done:
				# every episode update the target model to be same with model
				agent.update_target_model()

				# every episode, plot the play time
				episodes.append(e)
				print("episode:", e, "  score:", score, "  memory length:", len(agent.memory), "  epsilon:", agent.epsilon)
		agent.model.save("drone_ddqn.h5")
