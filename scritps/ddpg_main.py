#!/usr/bin/env python3

import sys
import pandas as pd
from ddpg import DDPG
from PlutoEnvDDPG import Drone_state
import numpy as np
import rospy

num_episodes = 1000

#target_pos = np.array([0., 0., 140.])
#task = Task(target_pos=target_pos)
task = Drone_state()
agent = DDPG(task)
best_score = -1000
best_x = 0
best_y = 0
best_z = 0
best_episode = 0
data = {}

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    rospy.sleep(1)
    score = 0
    
    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        score += reward
        
        if score > best_score:
            #best_x = task.sim.pose[0]
            #best_y = task.sim.pose[1]
            #best_z = task.sim.pose[2]
            best_episode = i_episode
        best_score = max(score, best_score)
        '''data[i_episode] = {'Episode': i_episode, 'Reward':score,'Action':action,'Best_Score':best_score,
                            'x':task.sim.pose[0],'y':task.sim.pose[1],'z':task.sim.pose[2]}'''
        if done:
            '''print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), last_position = ({:5.1f},{:5.1f},{:5.1f}), best_position = ({:5.1f},{:5.1f},{:5.1f})".format(
                i_episode, score, best_score, task.sim.pose[0], task.sim.pose[1], task.sim.pose[2], best_x, best_y, best_z), end="")'''
            print("Episode:", i_episode, "Score:", score)    
            break
    sys.stdout.flush()
