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

prev_z = 30

z_hold = 30


class Drone_state(object):
	
	def __init__(self):
		self.z=0.0
		self.drone_state = 30.0
		self.done = False
		self.is_reset = False
		self.Q = dict()
		self.epsilon = 1.0
		self.epsilon_max = 1.0
		self.epsilon_min = 0.01
		self.alpha = 0.1
		self.gamma = 0.8
        	
        	
	def build_state(self, state):
		#state = str(round(state,2))
		print(state)
		state = str(int(round(state)))
		return state	
        	
        	
	def get_maxQ(self, state):
		maxQ = -10000000
		for action in self.Q[state]:
			if self.Q[state][action] > maxQ:
				maxQ = self.Q[state][action]
		return maxQ
		
		
	def createQ(self, state):
		if state not in self.Q.keys():
			self.Q[state] = self.Q.get(state, copy.deepcopy(act_space))
		return            		
		
	
	def choose_action(self, state):
		if random.random() < self.epsilon:
			action = random.choice(val_act)
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
		self.Q[state][action] = (1 - self.alpha)*self.Q[state][action] + self.alpha*(reward + (self.gamma*maxQ_next_state))
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
		global prev_z 
		current_state = env.get_drone_state()
		print("current : ", current_state)
		if(action[0]=='u'):
			throttle(int(action[1:]))
			rospy.sleep(0.04)
		elif(action[0]=='d'):
			throttle(int(action[1:]))
			rospy.sleep(0.04)				
                		
		next_state = env.get_drone_state()
		prev_z = next_state
		print("Next : ", next_state)
		reward = get_reward(next_state)
		print("reward : ", reward)
		
		if(next_state<22 or next_state>38): #or abs(xp)>7 or abs(yp)>6):
			env.done = True

		return next_state, reward, env.done	
        	

env = Drone_state()

act_space=dict()

val_act = []

def create_act_space():
	for j in range(1500,1701,10):
		a='u'	
		act_space[a+str(j)] = act_space.get(a+str(j),0.0)
		val_act.append(a+str(j))
		
	for j in range(1300,1500,10):
		a='d'	
		act_space[a+str(j)] = act_space.get(a+str(j),0.0)
		val_act.append(a+str(j))	
	random.shuffle(val_act)
	print(val_act)		
						
					        	

def get_reward(state):
	reward = 0.0
	if ((abs(state - z_hold) < 1)) :
		# goal reached 
		reward = 100
		
	elif (abs(state - z_hold) < 2) :
		reward = 20
		
	elif (abs(state - z_hold) < 3) :
		reward = 10
		
	elif (abs(state - z_hold) < 4) :
		reward = 5
	
	elif (abs(state - z_hold) < 5) :
		reward = -5
		
	elif (abs(state - z_hold) < 6) :
		reward = -10				
	
	elif(state<22 or state>38):
		reward = -100										
					
	else:
		reward = -50
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
	#rospy.sleep(1)	
    	
    	
def cb_drone_state(pose):
	#n = int(round(pose.poses[0].position.z))
	env.set_drone_state(pose.poses[0].position.z)   #set z value obtained in bracket
    
def subscriber():
	rospy.Subscriber('whycon/poses', PoseArray, cb_drone_state) 
	
def save_Q(Q_table):
	with open('Qtable.csv', 'w') as csv_file:
		writer = csv.writer(csv_file)
		acti=copy.deepcopy(val_act)
		acti.insert(0,'act')
		writer.writerow(acti)
		for key, value in Q_table.items():
			act=[]
			act.append(key)	  
			for vals in val_act:
				act.append(Q_table[key][vals])
			writer.writerow(act)				

num_episodes = 6000

#tf.reset_default_graph()

subscriber()	 

create_act_space()  


      
while not rospy.is_shutdown():
	memory_states = []
	memory_targets = []
	for _ in range(num_episodes):
		state = env.reset()
		rospy.sleep(1)
		cmd.rcAUX1=1500
		prev_z = 30
		env.done = False
		total_reward = 0
		while env.done == False:
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
				env.epsilon = 1
			else:	
				env.epsilon = env.epsilon_min + (env.epsilon_max - env.epsilon_min)*(math.exp(-0.01*(_-1000)))
                    
			#take an e-greedy action
			next_state, reward, done = env.step(action)
            
			total_reward += reward

			next_state_temp = env.build_state(next_state)
			env.createQ(next_state_temp)
            
			env.learn(state, action, reward, next_state_temp)

			state = next_state
		save_Q(env.Q)	
		print(next_state)
		print ("reward in episode ",_," is: ",total_reward) 
		
	#validation for 100 episodes
	for _ in range(0,100):
		state = str(int(round(env.reset())))
		cmd.rcAUX1=1500
		env.done = False
		total_reward = 0
		while env.done == False:
			env.createQ(state)	
			action = env.choose_action(state)
			env.epsilon = 0.00001
                    	#take an e-greedy action
			next_state, reward, done = env.step(action)
            		total_reward += reward
			state = next_state
		save_Q(env.Q)	
		print ("reward in episode ",_," is: ",total_reward) 	    	
	
	
				 	
		
		
		
		
		
