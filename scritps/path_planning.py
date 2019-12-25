#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import gym
import random
import copy
import math

import roslib
import rospy
import random
import time
import math

#Ros packages
from plutodrone.msg import *
#from pid_tune.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int32
from std_msgs.msg import Int16
from std_msgs.msg import Int64
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
import rospy
import time


rospy.init_node('pluto_fly', disable_signals = True)

pluto_cmd = rospy.Publisher('/drone_command', PlutoMsg, queue_size=10)
		
cmd = PlutoMsg()

cmd.rcRoll = 1500
cmd.rcPitch = 1500
cmd.rcYaw = 1500
cmd.rcThrottle = 1000
cmd.rcAUX1 = 1500
cmd.rcAUX2 = 1500
cmd.rcAUX3 = 1500
cmd.rcAUX4 = 1000
cmd.plutoIndex = 0


class DroneState(object):
	def __init__(self):
		self.x = 0.0
		self.y = 0.0
		self.z = 30.0
		self.drone_state = [self.x, self.y, self.z]
		self.done = False
		self.is_reset = False
		self.Q = dict()
		self.epsilon = 1.0
		self.epsilon_max = 1.0
		self.epsilon_min = 0.01
		self.alpha = 0.8
		self.gamma = 0.9

	def build_state(self, state):

		state = str(int(round(state[0])))+'_'+str(int(round(state[1])))
		return state

	def get_maxQ(self, state):
		maxQ = -10000000
		for action in self.Q[state]:
			if self.Q[state][action] > maxQ:
				maxQ = self.Q[state][action]
		return maxQ 

	def createQ(self, state):
		if state not in self.Q.keys():
			self.Q[state] = self.Q.get(state, {'0':0.0, '1':0.0, '2':0.0, '3':0.0})
		return

	def choose_action(self, state):
		valid_actions = ['0', '1', '2', '3']
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
		self.drone_state = [state[0], state[1], state[2]]

	def get_drone_state(self):
		return self.drone_state

	def reset(self):
		cmd.rcAUX1=1800
		pluto_cmd.publish(cmd)
		return env.get_drone_state()

	def step(self, action):
		current_state = env.get_drone_state()
		current_state_x = int(round(current_state[0]))
		current_state_y = int(round(current_state[1]))
		current_state_z = int(current_state[2])
		goal_state_x = 0
		goal_state_y = current_state_y
		goal_state_z = 30.0

		#pid.last_error_x = 0.0
		#pid.last_error_y = 0.0
		#pid.last_error_z = 0.0
		#pid.prev_errors = [0,0,0,0]

		if action == 0:
			goal_state_x = current_state_x + 1
			goal_state_y = current_state_y
			while True:
				pitch, roll, thr = pid.run(current_state_x, goal_state_x, current_state_y, goal_state_y, current_state_z, goal_state_z)
				#publish action                
				cmd.rcPitch = pitch
				cmd.rcRoll = roll
				cmd.rcThrottle = thr
				pluto_cmd.publish(cmd)
				if abs(goal_state_x - current_state_x ) <= 0.05 and abs(goal_state_y - current_state_y) <= 0.05 and abs(goal_state_z - current_state_z) <= 0.05:
					break
				current_state = env.get_drone_state()
				current_state_x = current_state[0]
				current_state_y = current_state[1]
				current_state_z = current_state[2]
				rospy.sleep(0.03)

		if action == 1:
			goal_state_x = current_state_x - 1
			goal_state_y = current_state_y
			while True:
				pitch, roll, thr = pid.run(current_state_x, goal_state_x, current_state_y, goal_state_y, current_state_z, goal_state_z)
				#publish action
				cmd.rcPitch = pitch
				cmd.rcRoll = roll
				cmd.rcThrottle = thr
				pluto_cmd.publish(cmd)
				if abs(goal_state_x - current_state_x ) <= 0.05 and abs(goal_state_y - current_state_y) <= 0.05 and abs(goal_state_z - current_state_z) <= 0.05:
					break
				current_state = env.get_drone_state()
				current_state_x = current_state[0]
				current_state_y = current_state[1]
				current_state_z = current_state[2]
				rospy.sleep(0.03)
			

		if action == 2: 
			goal_state_y = current_state_y + 1
			goal_state_x = current_state_x
			while True:
				pitch, roll, thr = pid.run(current_state_x, goal_state_x, current_state_y, goal_state_y, current_state_z, goal_state_z)
				#publish action
				cmd.rcPitch = pitch
				cmd.rcRoll = roll
				cmd.rcThrottle = thr
				pluto_cmd.publish(cmd)			                
				if abs(goal_state_y - current_state_y ) <= 0.05 and abs(goal_state_x - current_state_x ) <= 0.05 and abs(goal_state_z - current_state_z) <= 0.05:   
					break
				current_state = env.get_drone_state()
				current_state_x = current_state[0]
				current_state_y = current_state[1]
				current_state_z = current_state[2]
				rospy.sleep(0.03)

		if action == 3: 
			goal_state = current_state_y - 1
			goal_state_x = current_state_x
			while True:
				pitch, roll, thr = pid.run(current_state_x, goal_state_x, current_state_y, goal_state_y, current_state_z, goal_state_z)
				#publish action
				cmd.rcPitch = pitch
				cmd.rcRoll = roll
				cmd.rcThrottle = thr
				pluto_cmd.publish(cmd)
				if abs(goal_state_y - current_state_y ) <= 0.05 and abs(goal_state_x - current_state_x ) <= 0.05 and abs(goal_state_z - current_state_z) <= 0.05:   
					break
				current_state = env.get_drone_state()
				current_state_x = current_state[0]
				current_state_y = current_state[1]
				current_state_z = current_state[2]
				rospy.sleep(0.03)

		next_state = env.get_drone_state()
		reward = get_reward(next_state)
        
		if (abs(next_state[0]) > 7 or abs(next_state[1]) > 6 or reward == 100):
			env.done = True

		return next_state, reward, env.done

class PIDController(object):
	def __init__(self):
	
		self.alt_error = rospy.Publisher('/alt_error', Float64, queue_size=1)
		self.pitch_error = rospy.Publisher('/pitch_error', Float64, queue_size=1)
		self.roll_error = rospy.Publisher('/roll_error', Float64, queue_size=1)
		#self.yaw_error = rospy.Publisher('/yaw_error', Float64, queue_size=1)
		self.zero_line = rospy.Publisher('/zero_error_line',Float64, queue_size=1)
		self.fifteen = rospy.Publisher('/fiftenn',Float64,queue_size=1)

		

		# Subscribing to /whycon/poses, /drone_yaw
		#rospy.Subscriber('/drone_yaw', Float64, self.drone_yaw)
		rospy.Subscriber('/pid_tuning_altitude',PidTune,self.altitude_set_pid)
		rospy.Subscriber('/pid_tuning_pitch',PidTune,self.pitch_set_pid)
		rospy.Subscriber('pid_tuning_roll',PidTune,self.roll_set_pid)
		self.drone_position = [0.0,0.0,0.0]	#current position of drone
		self.setpoint = [0.0,0.0,22.0]
		
		self.Kp = [7.2,10.32,56.4,0]
		self.Kd = [487.2,509.7,38.4,0]
		self.Ki = [0.024,0.016,0,0]
        
		self.prev_errors = [0.0,0.0,0.0,0.0]
		self.max_values = [1800,1800,2000,1800]
		self.min_values = [1200,1200,1200,1200]
		self.pid_values = [0,0,0,0]
		self.errors = [0,0,0,0]
		self.errsum = [0,0,0,0]
		self.lastTime = 0.0
		self.key_value = 0
        
	def pitch_set_pid(self,pitch):
		self.Kp[1] = pitch.Kp * 1    #0.06 
		self.Ki[1] = pitch.Ki * 1    #0.001
		self.Kd[1] = pitch.Kd * 1    #0.3

	def roll_set_pid(self,roll):
		self.Kp[0] = roll.Kp * 1     #0.06 
		self.Ki[0] = roll.Ki * 1     #0.001
		self.Kd[0] = roll.Kd * 1     #0.3
		
	def altitude_set_pid(self,alt):
		self.Kp[2] = alt.Kp * 1 #0.06 # This is just for an example. You can change the fraction value accordingly
		self.Ki[2] = alt.Ki * 1 #0.001
		self.Kd[2] = alt.Kd * 1 #0.3	

	def run(self, current_x, goal_x, current_y, goal_y, current_z, goal_z):
		self.drone_position = [current_x, current_y, current_z]
		self.setpoint = [goal_x, goal_y, goal_z]    
		self.errors[0] = self.setpoint[0] - self.drone_position[0]
		self.errors[1] = self.setpoint[1] - self.drone_position[1]
		self.errors[2] = self.setpoint[2] - self.drone_position[2]
		self.errsum[0] = (self.errsum[0] + self.errors[0])
		self.errsum[1] = (self.errsum[1] + self.errors[1])
		self.errsum[2] = self.errsum[2] + self.errors[2]
		self.pid_values[0] = (self.Kp[0] * self.errors[0]) + (self.Ki[0] * self.errsum[0]) + (self.Kd[0] * (self.errors[0] - self.prev_errors[0]))
		self.pid_values[1] = (self.Kp[1] * self.errors[1]) + (self.Ki[1] * self.errsum[1]) + (self.Kd[1] * (self.errors[1] - self.prev_errors[1]))
		self.pid_values[2] = (self.Kp[2] * self.errors[2]) + (self.Ki[2] * self.errsum[2]) + (self.Kd[2] * (self.errors[2] - self.prev_errors[2]))
		cmd.rcThrottle = 1500 - self.pid_values[2]
		cmd.rcRoll = 1500 + self.pid_values[0]
		cmd.rcPitch = 1500 - self.pid_values[1]
		if cmd.rcRoll > 1700:
			cmd.rcRoll = 1700

		if cmd.rcPitch > 1700:
			cmd.rcPitch = 1700
			
		if cmd.rcThrottle > 1800:
			cmd.rcThrottle = 1800

		if cmd.rcRoll < 1300:
			cmd.rcRoll = 1300

		if cmd.rcPitch < 1300:
			cmd.rcPitch = 1300
				
		if cmd.rcThrottle < 1200:
			cmd.rcThrottle = 1200
			
		self.roll_error.publish(self.errors[0])
		self.pitch_error.publish(self.errors[1])
		self.alt_error.publish(self.errors[2])
		self.zero_line.publish(0)
		self.fifteen.publish(1500)
		self.prev_errors[0] = self.errors[0]
		self.prev_errors[1] = self.errors[1]
		self.prev_errors[2] = self.errors[2]			
        
		return int(round(cmd.rcRoll)), int(round(cmd.rcPitch)), int(round(cmd.rcThrottle))

env = DroneState()
pid = PIDController()

def get_reward(state):
    reward = 0.0
    if abs(state[0] - 5.0) <= 0.05 and abs(state[1] - 5.0) <= 0.05:
        # goal reached 
        reward = 100
    else:
        reward = -1
    # if (state[0] < -0.05 or state[0] > 5.05 or state[1] < -0.05 or state[1] > 5.05):
    return reward
    
def arm():
	cmd.rcAUX4 = 1500
	cmd.rcThrottle = 1000
	pluto_cmd.publish(cmd)
	rospy.sleep(.1)    


def cb_drone_state(pose):
    env.set_drone_state([pose.poses[0].position.x, pose.poses[0].position.y, pose.poses[0].position.z])


def subscriber():
	rospy.Subscriber('whycon/poses', PoseArray, cb_drone_state)


num_episodes = 10000

tf.reset_default_graph()

subscriber()

while not rospy.is_shutdown():
    memory_states = []
    memory_targets = []
    for _ in range(num_episodes):
        state = env.reset()
        cmd.rcAUX1 = 1500
        rospy.sleep(1)
        arm()
        env.done = False
        total_reward = 0
        rospy.sleep(0.25)
        while env.done == False:
            #print i
            state = env.build_state(state)
            env.createQ(state)
            #print state
            #ep_states.append(state)
            #print memory_states
            print ("Q Table :", env.Q)
            action = env.choose_action(state)
            env.epsilon = env.epsilon_min + (env.epsilon_max - env.epsilon_min)*(math.exp(-0.01*_))
                    
            #take an e-greedy action
            next_state, reward, done = env.step(int(action))
            
            total_reward += reward

            next_state_temp = env.build_state(next_state)
            env.createQ(next_state_temp)
            
            env.learn(state, action, reward, next_state_temp)

            state = next_state

        print ("reward in episode ",_," is: ",total_reward)






