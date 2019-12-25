#!/usr/bin/env python

'''
* Functions: arm() , disarm() , whycon_callback() , drone_yaw() , indentify_key() , pid()
* Global Variables: self.drone_position , self.setpoint , self.cmd.rcRoll , self.cmd.rcPitch , self.cmd.rcYaw , self.cmd.rcThrottle , self.cmd.rcAUX1 , self.cmd.rcAUX2 , self.cmd.rcAUX3 , self.cmd.rcAUX4 , self.Kp , self.Ki , self.Kd , self.prev_errors , self.max_values , self.min_values , self.pid_values , self.errors , self.errsum , self.lastTime , self.key_value , self.sample_time , self.command_pub , self.alt_error , self.pitch_error , self.roll_error , self.yaw_error , self.zero_line , self.fifteen

'''

# Importing the required libraries

from plutodrone.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int16
from std_msgs.msg import Int64
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
import rospy
import time


class Edrone():
	"""docstring for Edrone"""
	def __init__(self):
		
		rospy.init_node('drone_control')	# initializing ros node with name drone_control
		self.drone_position = [0.0,0.0,0.0,0.0]	#current position of drone

		self.setpoint = [-7.0,-3.0,22.0,0.0] 

		#Declaring a cmd of message type PlutoMsg and initializing values
		self.cmd = PlutoMsg()
		self.cmd.rcRoll = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcThrottle = 1500
		self.cmd.rcAUX1 = 1500
		self.cmd.rcAUX2 = 1500
		self.cmd.rcAUX4 = 1500



		#initial setting of Kp, Kd and ki for [roll, pitch, throttle, yaw]
		self.Kp = [0,0,56,0]
		self.Ki = [0.024,0.016,0,0]
		self.Kd = [487.2,509.7,38.4,0]


		#required variables for pid 
		self.prev_errors = [0,0,0,0]
		self.max_values = [1800,1800,2000,1800]
		self.min_values = [1200,1200,1200,1200]
		self.pid_values = [0,0,0,0]
		self.errors = [0,0,0,0]
		self.errsum = [0,0,0,0]
		self.lastTime = 0.0
		self.key_value = 0


		self.sample_time = 0.030 #sample time in which pid need to run# in seconds


		# Publishing /drone_command, /alt_error, /pitch_error, /roll_error, /yaw_error
		self.command_pub = rospy.Publisher('/drone_command', PlutoMsg, queue_size=1)
		self.alt_error = rospy.Publisher('/alt_error', Float64, queue_size=1)
		self.pitch_error = rospy.Publisher('/pitch_error', Float64, queue_size=1)
		self.roll_error = rospy.Publisher('/roll_error', Float64, queue_size=1)
		self.yaw_error = rospy.Publisher('/yaw_error', Float64, queue_size=1)
		self.zero_line = rospy.Publisher('/zero_error_line',Float64, queue_size=1)
		self.fifteen = rospy.Publisher('/fiftenn',Float64,queue_size=1)
		

		# Subscribing to /whycon/poses, /drone_yaw
		#rospy.Subscriber('/input_key', Int16, self.indentify_key )
		rospy.Subscriber('whycon/poses', PoseArray, self.whycon_callback)
		rospy.Subscriber('/drone_yaw', Float64, self.drone_yaw)
		rospy.Subscriber('/pid_tuning_altitude',PidTune,self.altitude_set_pid)
		rospy.Subscriber('/pid_tuning_pitch',PidTune,self.pitch_set_pid)
		rospy.Subscriber('pid_tuning_roll',PidTune,self.roll_set_pid)

		#self.arm() 

	'''

	* Function Name: disarm()
	* Input: None
	* Output: returns or publishes self.cmd.rcAUX4 value
	* Logic: this function disarms the drone by publishing or providing it with a value of self.cmd.rcAUX4 = 1100
	* Example Call: if self.key_value == 0:         
				self.disarm()

	'''

	# Disarming condition of the drone
	def disarm(self):
		self.cmd.rcAUX4 = 1100
		self.command_pub.publish(self.cmd)
		rospy.sleep(1)
		print "Disarm"


	'''

	* Function Name: arm()
	* Input: None
	* Output: returns or publishes self.cmd.rcRoll , self.cmd.rcYaw , self.cmd.rcPitch , self.cmd.rcThrottle , self.cmd.rcAUX4 values
	* Logic: this function arms the drone by publishing or providing it with a value of self.cmd.rcRoll = 1500 , self.cmd.rcYaw = 1500 , self.cmd.rcPitch = 1500 , self.cmd.rcThrottle = 1000 , self.cmd.rcAUX4 = 1500
	* Example Call: if self.key_value == 70:
			self.disarm()
			self.arm()

	'''

	def arm(self):   # Arming condition of the drone

		#self.disarm()

		self.cmd.rcRoll = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcThrottle = 1000
		self.cmd.rcAUX4 = 1500
		self.command_pub.publish(self.cmd)	# Publishing /drone_command
		print "Arm"
		rospy.sleep(1)


	'''

	* Function Name: whycon_callback()
	* Input: msg has the cordinates of whycon marker detected by camera and it is in whycon/poses rostopic
	* Output: cordinates of each axis is been stored in separate variables
	* Logic: cordinates of each axis is been accessed fron whycon/poses rostopic into separate variables
	* Example Call: rospy.Subscriber('whycon/poses', PoseArray, self.whycon_callback)

	'''
	# Whycon callback function
	# The function gets executed each time when /whycon node publishes /whycon/poses 
	def whycon_callback(self,msg):
		self.drone_position[0] = msg.poses[0].position.x
		self.drone_position[1] = msg.poses[0].position.y
		self.drone_position[2] = msg.poses[0].position.z
		

	def altitude_set_pid(self,alt):
		self.Kp[2] = alt.Kp * 0.06 # This is just for an example. You can change the fraction value accordingly
		self.Ki[2] = alt.Ki * 0.001
		self.Kd[2] = alt.Kd * 0.3

	#----------------------------Define callback function like altitide_set_pid to tune pitch, roll and yaw as well--------------
	
	def pitch_set_pid(self,pitch):
		self.Kp[1] = pitch.Kp * 0.06 
		self.Ki[1] = pitch.Ki * 0.001
		self.Kd[1] = pitch.Kd * 0.3

	def roll_set_pid(self,roll):
		self.Kp[0] = roll.Kp * 0.06 
		self.Ki[0] = roll.Ki * 0.001
		self.Kd[0] = roll.Kd * 0.3

	'''

	* Function Name: drone_yaw()
	* Input: yaw_s has the current yaw value of drone in /drone_yaw rostopic
	* Output: drone yaw is been stored as 4th drone position value in list
	* Logic: drone yaw value is been accessed from /drone_yaw rostopic into separate variables
	* Example Call: rospy.Subscriber('/drone_yaw', Float64, self.drone_yaw)

	'''

	def drone_yaw(self,yaw_s):
		self.drone_position[3] = yaw_s.data

	'''

	* Function Name: indentify_key()
	* Input: msg has the key value
	* Output: drone is armed disarmed and pid function is called
	* Logic: as per the key value received different function is been called
	* Example Call: rospy.Subscriber('/input_key', Int16, self.indentify_key )

	'''

	'''

	* Function Name: pid()
	* Input: none
	* Output: drone is been controlled using pid algorithm by publishing throttle,roll,pitch values of drone
	* Logic: drone cordinates error or difference from its setpoint is been used to handle the drone
	* Example Call: if self.key_value == 100:
			self.pid()

	'''

	def pid(self):
		while True:
			zero = 0
			fif = 1500
			now = time.time()
			timechange = now-self.lastTime
			if timechange >= self.sample_time:

				self.errors[0] = self.setpoint[0] - self.drone_position[0]

				self.errors[1] = self.setpoint[1] - self.drone_position[1]
			
				self.errors[2] = self.setpoint[2] - self.drone_position[2]

				self.errsum[0] = (self.errsum[0] + self.errors[0])

				self.errsum[1] = (self.errsum[1] + self.errors[1])
			
				self.errsum[2] = self.errsum[2] + self.errors[2]

				self.pid_values[0] = (self.Kp[0] * self.errors[0]) + (self.Ki[0] * self.errsum[0]) + (self.Kd[0] * (self.errors[0] - self.prev_errors[0]))

				self.pid_values[1] = (self.Kp[1] * self.errors[1]) + (self.Ki[1] * self.errsum[1]) + (self.Kd[1] * (self.errors[1] - self.prev_errors[1]))
			
				self.pid_values[2] = (self.Kp[2] * self.errors[2]) + (self.Ki[2] * self.errsum[2]) + (self.Kd[2] * (self.errors[2] - self.prev_errors[2]))

				self.cmd.rcThrottle = 1500 - self.pid_values[2]

				self.cmd.rcRoll = 1500 + self.pid_values[0]

				self.cmd.rcPitch = 1500 - self.pid_values[1]
			
				

				if self.cmd.rcRoll > self.max_values[0]:
					self.cmd.rcRoll = self.max_values[0]

				if self.cmd.rcPitch > self.max_values[1]:
					self.cmd.rcPitch = self.max_values[1]
			
				if self.cmd.rcThrottle > self.max_values[2]:
					self.cmd.rcThrottle = self.max_values[2]

				if self.cmd.rcRoll < self.min_values[0]:
					self.cmd.rcRoll = self.min_values[0]

				if self.cmd.rcPitch < self.min_values[1]:
					self.cmd.rcPitch = self.min_values[1]
				
				if self.cmd.rcThrottle < self.min_values[2]:
					self.cmd.rcThrottle = self.min_values[2]
				
				self.command_pub.publish(self.cmd)

				self.roll_error.publish(self.errors[0])

				self.pitch_error.publish(self.errors[1])
			
				self.alt_error.publish(self.errors[2])
			
				self.zero_line.publish(zero)
			
				self.fifteen.publish(fif)

				self.prev_errors[0] = self.errors[0]

				self.prev_errors[1] = self.errors[1]
			
				self.prev_errors[2] = self.errors[2]
			
				self.lastTime = now
				
				print "PID"


if __name__ == '__main__':

	e_drone = Edrone()
	e_drone.disarm()
	rospy.sleep(1)
	e_drone.arm()
	while not rospy.is_shutdown():
		e_drone.pid()
		rospy.spin()
