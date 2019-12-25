#!/usr/bin/env python3

import rospy
import random
#import gym
import math
import numpy as np
import random
import copy
import csv
import tensorflow
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#Ros packages
from plutodrone.msg import *
#from pid_tune.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int32
import rospy
import time

class Drone_state():
	def __init__(self):
		self.act_spc=[]
		for i in range(0,41):
			self.act_spc.append(i)
		random.shuffle(self.act_spc)
		myFile = open('actions.csv', 'w')  
		with myFile:  
			writer = csv.writer(myFile)
			writer.writerow(self.act_spc)
		myFile.close()		
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
		self.drone_state = [0.0, 0.0]
			
	def action_space(self):
		return self.act_spc
		
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
		#print(action)	
		self.done = False
		current_state = self.get_drone_state()
		#print("Current state : ", current_state)
		self.throttle(action)
		rospy.sleep(0.04)						
		next_state = self.get_drone_state()
		#print("Next state : ", next_state)
		self.prev_z = next_state[0]
		reward = self.get_reward(next_state)
		if(next_state[0]<22 or next_state[0]>38): #or abs(xp)>7 or abs(yp)>6):
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
	
		if(state[1] < 10):
			reward = reward + 20
		elif(abs(state[1]) >= 10 and abs(state[1]) < 20):
			reward = reward + 5
		elif(abs(state[1]) >= 20 and abs(state[1]) < 30):
			reward = reward - 5
		elif(abs(state[1]) >= 60):
			reward = reward - 50
		else:
			reward = reward - 10				
		return reward		
		
	def set_drone_state(self, state):
		self.drone_state = state

	def get_drone_state(self):
		return self.drone_state	
		
	def cb_drone_state(self,pose):
		n=pose.poses[0].position.z
		nv = (self.prev_z - n)/0.05
		self.set_drone_state([n, nv])   #set z value obtained in bracket	
		
			

class DQNCartPoleSolver():
	def __init__(self, n_episodes=20, n_win_ticks=195, max_env_steps=None, gamma=0.8, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.9, batch_size=64, monitor=False, quiet=False):
		self.memory = deque(maxlen=100000)
		self.env = Drone_state()
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_max = 1.0
		self.epsilon_decay = epsilon_log_decay
		self.alpha = alpha
		self.alpha_decay = alpha_decay
		self.n_episodes = n_episodes
		self.n_win_ticks = n_win_ticks
		self.batch_size = batch_size
		self.quiet = quiet
		#if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
		# Init model
		self.model = Sequential()
		self.model.add(Dense(150, input_dim=2, activation='tanh'))
		self.model.add(Dense(300, activation='selu'))
		self.model.add(Dense(400, activation='tanh'))
		self.model.add(Dense(41, activation='linear'))
		self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
		print("Entered drone state")
	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def choose_action(self, state, epsilon):
		#return random.choice(self.env.action_space()) if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))
		if (np.random.random() <= epsilon):
			action = random.choice(self.env.action_space())
			print("random : ", action)
		else:
			action = np.argmax(self.model.predict(state))
			print("greedy : ", action)
		return action	
			
	def get_epsilon(self, t, r):
		#return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))
		if(t<500):
			ep = 1
		else : 
			ep = self.epsilon_min + (self.epsilon_max - self.epsilon_min)*(math.exp(-0.01*(t-500)))
		return ep		

	def preprocess_state(self, state):
		return np.reshape(state,[1,2])

	def replay(self, batch_size):
		for i in range(0,1000):
			x_batch, y_batch = [], []
			minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
			for state, action, reward, next_state, done in minibatch:
				y_target = self.model.predict(state)
				y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
				x_batch.append(state[0])
				y_batch.append(y_target[0])
        
			self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
		for i in range(1,1000):
			self.memory.popleft()	
		#if self.epsilon > self.epsilon_min:
			#self.epsilon *= self.epsilon_decay

	def run(self):
		scores = deque(maxlen=100)
		print("Entered run")
		e=0
		#for e in range(self.n_episodes):
		while(e<3000):
			for k in range(0,50):
				state = self.preprocess_state(self.env.reset())
				rospy.sleep(1)
				done = False
				total_reward=0
				#i = 0
				while not done:
					action = self.choose_action(state, self.get_epsilon(e,k))
					next_state, reward, done = self.env.step(action)
					next_state = self.preprocess_state(next_state)
					self.remember(state, action, reward, next_state, done)
					state = next_state
					total_reward += reward
				#i += 1
				print("Total reward : ", total_reward)	
				print('Episode : ', e)
				e+=1
			#scores.append(i)
			#mean_score = np.mean(scores)
			#if mean_score >= self.n_win_ticks and e >= 100:
				#if not self.quiet: print('Ran {} episodes. Solved after {} trials'.format(e, e - 100))
				#return e - 100
			#if e % 100 == 0 and not self.quiet:
				#print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
			state = self.env.reset()
			self.replay(self.batch_size)
			self.model.save('Qnet.h5') 
			print("Episode End : ", e)
        
		if not self.quiet: print('Did not solve after {} episodes'.format(e))
		return e

if __name__ == '__main__':
	agent = DQNCartPoleSolver()
	agent.run()
