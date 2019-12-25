#!/usr/bin/env python3

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

z_hold = 30


class Drone_state(object):
	
	def __init__(self):
		self.z=0.0
		self.drone_state = self.z
		self.done = False
		self.is_reset = False
		self.Q = dict()
		self.epsilon = 1.0
		self.epsilon_max = 1.0
		self.epsilon_min = 0.01
		self.alpha = 0.4
		self.gamma = 0.9
        	
	
	def choose_action(self, state):
		if not(state in ls_states):
			ls_states.append(state)
			ls_max_val_act.append('u1500')
		idx = ls_states.index(state)
		action = ls_max_val_act[idx]
		return action	
		
	
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
		print("current : ", current_state)
		if(action[0]=='u'):
			throttle(int(action[1:]))
			rospy.sleep(0.04)
		elif(action[0]=='d'):
			throttle(int(action[1:]))
			rospy.sleep(0.04)				
                		
		next_state = int(round(env.get_drone_state()))
		print("Next : ", next_state)
		reward = get_reward(next_state)
		print("reward : ", reward)
		
		if(next_state<22 or next_state>38): #or abs(xp)>7 or abs(yp)>6):
			env.done = True

		return next_state, reward, env.done	
        	

env = Drone_state()	
						
					        	

def get_reward(state):
	reward = 0.0
	if ((abs(state - z_hold) < 1)) :
		# goal reached 
		reward = 100
		
	elif (abs(state - z_hold) < 2) :
		reward = 25
		
	elif (abs(state - z_hold) < 3) :
		reward = 10
		
	elif (abs(state - z_hold) < 4) :
		reward = 5
	
	elif (abs(state - z_hold) < 5) :
		reward = 1
		
	elif (abs(state - z_hold) < 6) :
		reward = -10				
	
	elif(state<23 or state>37):
		reward = -100										
					
	else:
		reward = -50
	# if (state[0] < -0.05 or state[0] > 5.05 or state[1] < -0.05 or state[1] > 5.05):
	return reward	
	
	
	
def throttle(t):
	cmd.rcThrottle = t
	pluto_cmd.publish(cmd)
	
	
def reset_publish():
	cmd.rcAUX1=1800
	pluto_cmd.publish(cmd)
	#rospy.sleep(1)	
    	
    	
def cb_drone_state(pose):
	env.set_drone_state(pose.poses[0].position.z)   #set z value obtained in bracket
    
def subscriber():
	rospy.Subscriber('whycon/poses', PoseArray, cb_drone_state) 				

num_episodes = 20000

subscriber()	   

ls_act=[]
ls_states=[]
ls_max_val_act=[]

with open('Qtable.csv','rt') as f:
	data = csv.reader(f)
	for row in data:
		ls_act = row
		ls_act=ls_act[1:]
		break
	i=0
	for row in data:
		ls_states.append(int(row[0]))
		ls_vals=row[1:]
		maxval=ls_vals[0]
		max_val_act=ls_act[0]
		for z in range(1,len(ls_vals)):
			if ls_vals[z]>maxval:
				maxval=ls_vals[z]
				max_val_act = ls_act[z]
		ls_max_val_act.append(max_val_act)
		#i+=1						
	f.close()
	
#print(ls_act)
print(ls_states)
print(ls_max_val_act)


while not rospy.is_shutdown():
	memory_states = []
	memory_targets = []
	#validation for 100 episodes
	for _ in range(0,100):
		state = int(round(env.reset()))
		rospy.sleep(1)
		print(state)
		cmd.rcAUX1=1500
		env.done = False
		total_reward = 0
		while env.done == False:	
			action = env.choose_action(state)
			env.epsilon = 0.00001
                    	#take an e-greedy action
			next_state, reward, done = env.step(action)
			total_reward += reward
			state = next_state	
		print ("reward in episode ",_," is: ",total_reward) 

