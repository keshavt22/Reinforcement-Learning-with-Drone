#!/usr/bin/env python3

import rospy
import random
#import gym
import math
import numpy as np
import random
import copy

#Ros packages
from plutodrone.msg import *
#from pid_tune.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int32
import rospy
import time

class Drone_state(object):
	def __init__(self):
		self.act_spc=[]
		for i in range(0,41):
			self.act_spc.append(i)
		random.shuffle(self.act_spc)
		#myFile = open('actions.csv', 'w')  
		#with myFile:  
			#writer = csv.writer(myFile)
			#writer.writerow(self.act_spc)
		#myFile.close()		
		self.done = False
		rospy.init_node('pluto_fly', disable_signals = True)
		rospy.Subscriber('whycon/poses', PoseArray, self.cb_drone_state)
		self.pluto_cmd = rospy.Publisher('/drone_command', PlutoMsg, queue_size=10)
		self.cmd = PlutoMsg()
		self.cmd.rcRoll = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcThrottle = 1000
		self.cmd.rcAUX1 = 1500
		self.cmd.rcAUX2 = 1500
		self.cmd.rcAUX3 = 1500
		self.cmd.rcAUX4 = 1000
		self.cmd.plutoIndex = 0	
		self.z_hold = 30.12
		self.prev_z = 30.12
		self.drone_state = [0.0, 0.0]
		self.state_size = 2
		self.action_low = 1300
		self.action_high = 1700
		self.action_size = 1
		
		
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
		#action = 1300 + (action*10)
		#action = int(round((200*action[0])+1500))
		action = int(round(action[0]))
		print('Action in env : ', action)	
		self.done = False
		current_state = self.get_drone_state()
		#print("Current state : ", current_state)
		self.throttle(action)
		rospy.sleep(0.05)						
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
			reward = 100
		elif (abs(state[0] - self.z_hold) < 2) :
			reward = 10
		elif (abs(state[0] - self.z_hold) < 3) :
			reward = 5
		elif (abs(state[0] - self.z_hold) < 4) :
			reward = -10		
		elif(state[0]<22 or state[0]>38):
			reward = -100										
		else:
			reward = -50
							
		return reward		
		
	def set_drone_state(self, state):
		self.drone_state = state

	def get_drone_state(self):
		return self.drone_state	
		
	def cb_drone_state(self,pose):
		n=pose.poses[0].position.z
		nv = (self.prev_z - n)/0.05
		self.set_drone_state([n, nv])   #set z value obtained in bracket
