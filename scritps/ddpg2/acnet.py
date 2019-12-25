#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class Critic(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Critic, self).__init__()
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, output_size)

	def forward(self, state, action):
		"""
		Params state and actions are torch tensors
		"""
		x = torch.cat([state, action], 1)
		x = F.selu(self.linear1(x))
		x = F.selu(self.linear2(x))
		x = self.linear3(x)
		return x
		
#learning_rate = 3e-4		

class Actor(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, learning_rate = 0.001):
		super(Actor, self).__init__()
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, output_size)
        
	def forward(self, state):
		"""
		Param state is a torch tensor
		"""
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		x = torch.tanh(self.linear3(x))
		return x
