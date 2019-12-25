# Q-Learning and Q-table 

Q-learning is a model-free reinforcement learning algorithm which generates Q-table after it gets trained to act in any environment.
The quality of our policy will improve upon training and will keep on improving.

![state_action](https://user-images.githubusercontent.com/44941918/61121705-7d54e280-a4bd-11e9-92ec-a46d41117abf.jpg)

Q-table maps from different states to what actions to take using state
action values [state, action].
We then update and store our q-values after an episode. This q-table becomes a reference table for our agent to select the best action based on the q-value.

In Q-Learning, unlike SARSA, based on experiences the agent gets greedy gradually i.e. it always goes
for immediate maximum reward.To maintain exploration and exploitation strategy,
we use epsilon greedy policy.

Acting randomly is important because it allows the agent to explore and discover new states that otherwise may not be selected during the exploitation process. You can balance exploration/exploitation using epsilon (ε) and setting the value of how often you want to explore vs exploit
    
### Here are the 3 basic steps:    

Agent starts in a state takes an action and receives a reward.

Agent selects action by referencing Q-table with highest value (max) OR by random (epsilon, ε).

Update q-values to generate Q-table.

### Q-value is updated using bellman's equation:
![bellmans eqn](https://user-images.githubusercontent.com/44941918/61126060-1b01df00-a4c9-11e9-862e-f8fb41bc309d.jpg)


### How to implement :

make sure that Q-table which is to be implemented must be present in the catkin workspace in which the package is present.
