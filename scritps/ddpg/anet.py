#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

class anet(object):
	def __init__(self, sess, s_dim, a_dim, action_bound, tau):
		self.sess = sess
		self.s_dim = s_dim
		self.a_dim = a_dim
		self.action_bound = action_bound
		self.tau = tau
		self.lr = tf.placeholder(tf.float32)

		# actor network
		self.states, self.out, self.scaled_out, self.net = self.create_a_net()

		# initialize target_net
		self.target_net = self.net
        
		# target network
		self.target_states, self.target_out, self.target_scaled_out, self.target_net = self.create_a_target_net()

		# update target weight
		self.update_target = [self.target_net[i].assign(tf.multiply(self.tau, self.net[i]) + tf.multiply((1-self.tau), self.target_net[i])) for i in range(len(self.target_net))]

		# initialize Q gradients
		self.Q_gradients = tf.placeholder(tf.float32, [None, self.a_dim])
        
		# combine gradients, minus sign because of tensorflow does descent but here needs ascend
		self.actor_gradients = tf.gradients(self.scaled_out, self.net, -self.Q_gradients)

		# optimize
		self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.net))
            
            
	# 2 hidden layer_relu
	def create_a_net(self):
		layer1_size = 400
		layer2_size = 300
		states = tf.placeholder(tf.float32, [None, self.s_dim])
		aW1 = tf.Variable(tf.random_uniform([self.s_dim, layer1_size],-1/np.sqrt(self.s_dim),1/np.sqrt(self.s_dim)), name="aW1")
		aB1 = tf.Variable(tf.random_uniform([layer1_size],-1/np.sqrt(self.s_dim),1/np.sqrt(self.s_dim)), name="aB1")
		aW2 = tf.Variable(tf.random_uniform([layer1_size, layer2_size],-1/np.sqrt(layer1_size),1/np.sqrt(layer1_size)), name="aW2")
		aB2 = tf.Variable(tf.random_uniform([layer2_size],-1/np.sqrt(layer1_size),1/np.sqrt(layer1_size)), name="aB2")
		aW3 = tf.Variable(tf.random_uniform([layer2_size, self.a_dim],-3e-3,3e-3), name="aW3")
		aB3 = tf.Variable(tf.random_uniform([self.a_dim],-3e-3,3e-3), name="aB3")
		XX = tf.reshape(states, [-1, self.s_dim]) 
		Y1l = tf.matmul(XX, aW1) 
		Y1 = tf.nn.relu(Y1l+aB1)
		Y2l = tf.matmul(Y1, aW2)
		Y2 = tf.nn.relu(Y2l+aB2)
		Ylogits = tf.matmul(Y2, aW3) + aB3
		out = tf.tanh(Ylogits)
		#scaled_out = out
		scaled_out = tf.multiply(out, self.action_bound)
		return states, out, scaled_out, [aW1, aB1, aW2, aB2, aW3, aB3]
   
    
	# target net
	def create_a_target_net(self):
		states = tf.placeholder(tf.float32, [None, self.s_dim])
		aW1, aB1, aW2, aB2, aW3, aB3 = self.target_net
		XX = tf.reshape(states, [-1, self.s_dim]) 
		Y1l = tf.matmul(XX, aW1)
		Y1 = tf.nn.relu(Y1l+aB1)
		Y2l = tf.matmul(Y1, aW2)
		Y2 = tf.nn.relu(Y2l+aB2)
		Ylogits = tf.matmul(Y2, aW3) + aB3
		out = tf.tanh(Ylogits)
		#scaled_out = out
		scaled_out = tf.multiply(out, self.action_bound)
		return states, out, scaled_out, [aW1, aB1, aW2, aB2, aW3, aB3]
       
       
	def train(self, states, Q_gradients, i):
		learning_rate=0.0001
		self.sess.run(self.optimize, feed_dict={self.states: states,self.Q_gradients: Q_gradients, self.lr:learning_rate})


	def predict(self, states):
		return self.sess.run(self.scaled_out, feed_dict={self.states: states})

    
	def predict_target(self, states):
		return self.sess.run(self.target_scaled_out, feed_dict={self.target_states: states})

    
	def update_target_network(self):
		self.sess.run(self.update_target)

