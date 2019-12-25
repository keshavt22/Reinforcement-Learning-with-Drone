#!/usr/bin/env python

import numpy as np
#import tensorflow as tf
#import gym
import random
import copy
import math
import csv

#Ros packages
from plutodrone.msg import *
#from pid_tune.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int32
import rospy
import time

rospy.init_node('pluto_fly', disable_signals = True)


#rate = rospy.Rate(120)

pluto_cmd = rospy.Publisher('/drone_command', PlutoMsg, queue_size=10)
		
cmd = PlutoMsg()

cmd.rcRoll = 1500
cmd.rcPitch = 1500
cmd.rcYaw = 1500
cmd.rcThrottle = 1500
cmd.rcAUX1 = 1500
cmd.rcAUX2 = 1500
cmd.rcAUX3 = 1500
cmd.rcAUX4 = 1000
cmd.plutoIndex = 0

x_hold = 0
y_hold = 0
z_hold = 30

class Drone_state(object):
	
	def __init__(self):
		self.x = 0.0
		self.y = 0.0
		self.z=40.0
		self.drone_state = [self.x, self.y, self.z]
		self.done = False
		self.is_reset = False
		self.Q = dict()
		self.epsilon = 1.0
		self.epsilon_max = 1.0
		self.epsilon_min = 0.01
		self.alpha = 0.7
		self.gamma = 0.6
        	
        	
	def build_state(self, state):
		state = str(int(round(state[0])))+'_'+str(int(round(state[1])))+'_'+str(int(round(state[2])))
		return state	
        	
        	
	def get_maxQ(self, state):
		maxQ = -10000000
		for action in self.Q[state]:
			if self.Q[state][action] > maxQ:
				maxQ = self.Q[state][action]
		return maxQ
		
		
	def createQ(self, state):
		if state not in self.Q.keys():
			self.Q[state] = self.Q.get(state, act_space)
		return            		
		
	
	def choose_action(self, state):
		valid_actions = val_act
		if random.random() < self.epsilon:
			action = random.choice(valid_actions)
			print ("random: ", action)
		else:
			actions = []
			maxQ = self.get_maxQ(state)
			for action in self.Q[state]:
				if self.Q[state][action] == float(maxQ):
					actions.append(action)
			action = random.choice(actions)
			print ("greedy: ", action)
		return action
		
		
	def learn(self, state, action, reward, next_state):
		maxQ_next_state = env.get_maxQ(next_state)
		# self.Q[state][action] = self.Q[state][action] + self.alpha*(reward - self.Q[state][action])
		self.Q[state][action] = (1 - self.alpha)*self.Q[state][action] + self.alpha*(self.gamma*(reward + maxQ_next_state))
		return	
		
	
	def set_drone_state(self, state):
		self.drone_state = state

	def get_drone_state(self):
		return self.drone_state

	def reset(self):
		#Ros environment reset
		reset_publish()
		return env.get_drone_state()   
	
        	
	def step(self, action):
		current_state = env.get_drone_state()
		if action[0] == 't':
			throttle(int(action[1:]))
			rospy.sleep(0.02)
			current_state = env.get_drone_state()
		elif action[0] == 'p':
			pitch(int(action[1:]))
			rospy.sleep(0.018)
			current_state = env.get_drone_state()
		elif action[0] == 'r':
			roll(int(action[1:]))
			rospy.sleep(0.018)
			current_state = env.get_drone_state()	
                		
		next_state = env.get_drone_state()
		reward = get_reward(next_state)
        
        	if(next_state[2]<20 or abs(next_state[0])>7.5 or abs(next_state[1])>6.5):
			env.done = True

		return next_state, reward, env.done	
        	

env = Drone_state()

act_space=dict()

val_act = []


def create_act_space():
	for i in range(0,2):
		for j in range(1400,1601,10):
			if(i==0):
				a='p'
			elif(i==1):
				a='r'	
			act_space[a+str(j)] = act_space.get(a+str(j),0.0)
			val_act.append(a+str(j))
	for k in range(1300,1701,10):
		a='t'
		act_space[a+str(k)] = act_space.get(a+str(k),0.0)
		val_act.append(a+str(k))		


def get_reward(state):
	reward = 0.0
	if (abs(state[0] - x_hold) <= 1) and (abs(state[1] - y_hold) <= 1) and (abs(state[2] - z_hold) <= 1) :
		# goal reached 
		reward = 100
		
	elif (abs(state[0] - x_hold) <= 2) and (abs(state[1] - y_hold) <= 2) and (abs(state[2] - z_hold) <= 2) :
		# goal reached 
		reward = 75
		
	elif (abs(state[0] - x_hold) <= 3) and (abs(state[1] - y_hold) <= 3) and (abs(state[2] - z_hold) <= 3) :
		reward = 50
		
	elif (abs(state[0] - x_hold) <= 4) and (abs(state[1] - y_hold) <= 4) and (abs(state[2] - z_hold) <= 4) :
		reward = 25
	
	elif (abs(state[0] - x_hold) <= 5) and (abs(state[1] - y_hold) <= 5) and (abs(state[2] - z_hold) <= 5) :
		reward = 0			
			
	else:
		reward = -10
	# if (state[0] < -0.05 or state[0] > 5.05 or state[1] < -0.05 or state[1] > 5.05):
	return reward	
	
	
	
def throttle(t):
	cmd.rcThrottle = t
	pluto_cmd.publish(cmd)
	
def pitch(p):
	cmd.rcPitch = p
	pluto_cmd.publish(cmd)
	
def roll(r):
	cmd.rcRoll = r
	pluto_cmd.publish(cmd)
	
def reset_publish():
	cmd.rcAUX1=1800
	pluto_cmd.publish(cmd)
	rospy.sleep(1)	
	

def arm():
	cmd.rcAUX4 = 1500
	cmd.rcThrottle = 1000
	pluto_cmd.publish(cmd)
	rospy.sleep(.1)

def disarm():
	cmd.rcAUX4 = 1100
	pluto_cmd.publish(cmd)
	rospy.sleep(.1)
    	
    	
def cb_drone_state(pose):	
	env.set_drone_state([pose.poses[0].position.x, pose.poses[0].position.y, pose.poses[0].position.z])   #set z value obtained in bracket
    
def subscriber():
	rospy.Subscriber('whycon/poses', PoseArray, cb_drone_state)   
	

num_episodes = 10000

#tf.reset_default_graph()

arm()

subscriber()	 

create_act_space()  

#w = csv.writer(open("Q_table.csv", "w")) 
      
while not rospy.is_shutdown():
	memory_states = []
	memory_targets = []
	cmd.rcAUX1=1500
	arm()
	rospy.sleep(1)
	print("Armed")
	for _ in range(num_episodes):
		state = env.reset()
		cmd.rcAUX1=1500
		called=1
		arm()
		env.done = False
		total_reward = 0
		#time.sleep(5)
		while (env.done == False):
			#print i
			state = env.build_state(state)
			env.createQ(state)
			#print state
			#ep_states.append(state)
			#print memory_states
			#print ("Q Table :", env.Q)
			#for i in env.Q:
				#print(i)	
			action = env.choose_action(state)
			
			if(_<1000):
				env.epsilon = 1.2
			else:	
				env.epsilon = env.epsilon_min + (env.epsilon_max - env.epsilon_min)*(math.exp(-0.01*(_-1000)))
                    
			#take an e-greedy action
			next_state, reward, done = env.step(action)
			total_reward += reward

			next_state_temp = env.build_state(next_state)
			env.createQ(next_state_temp)
            
			env.learn(state, action, reward, next_state_temp)
	
			state = next_state
		print(next_state)
		print ("reward in episode ",_," is: ",total_reward)      	
	
	
				 	
		
		
		
		
		
