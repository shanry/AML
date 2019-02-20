# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 18:47:21 2017

@author: Sharey
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:43:32 2017

@author: Sharey
"""

"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.2
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
import math
import matplotlib.pyplot as plt
# Hyper Parameters
NUM_EPISODES = 220
MAX_STEPS = 20000
BATCH_SIZE = 300
LR = 0.002                   # learning rate
TARGET_REPLACE_ITER = 64   # target update frequency
GAMMA = 0.9                 # reward discount
MEMORY_CAPACITY = 3000
DECAY = 20
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(50, 50)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            actions_value = self.eval_net.forward(x)
            action = int ( torch.max(actions_value, 1)[1].data.numpy() )
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    def get_learning_rate(self,t):
        return max(0.1, min(0.5, 1.0 - math.log10((t+1)/DECAY)))
    def get_explore_rate(self,t):
        return max(0.01, min(1, 1.0 - math.log10((t+1)/DECAY)))

    def learn(self):
        
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int).tolist()))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.data.numpy()

dqn = DQN()
x_data = []
y_data = []
r_data = []
e_data = []
learn_num = 0

print('\nCollecting experience...')
for i_episode in range(NUM_EPISODES):
    s = env.reset()
    #env.render()
    ep_r = 0
    steps = 0
    loss_local = [0]
    EPSILON = dqn.get_explore_rate(i_episode)
    for t in range (MAX_STEPS):
        #env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold -0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians -0.5
        r = r1 + r2 
        steps += 1
        dqn.store_transition(s, a, r, s_)
        ep_r += r
        s = s_
        if dqn.memory_counter > MEMORY_CAPACITY:
            learn_num += 1
            loss = dqn.learn()
            loss_local.append(loss)
            if done:
                y_data.append(np.array(loss_local).mean())
                e_data.append(i_episode)
            
            if t == MAX_STEPS-1:
                y_data.append(np.array(loss_local).mean())
                e_data.append(i_episode)
    
        
        
        if done:
            r_data.append(ep_r)
            x_data.append(i_episode)
            print('Ep: ', i_episode,          \
                  '| Ep_r: ', round(ep_r, 2),  \
                  '| steps: ', steps)
            break
        
        if t == MAX_STEPS-1:
           x_data.append(i_episode)
           r_data.append(ep_r)
           print("Episode %d terminated after %f MAX steps" % (i_episode, t))
            
        
        
        
sum_reward = 0
print ("-----------test---------------")
NUM_TEST = 100
MAX_STEPS = 20000
EPSILON = 0
result = []
for episode in range(NUM_TEST):

    # Reset the environment
    s = env.reset()
    #env.render()
    t = 0
    for t in range (MAX_STEPS):
        t += 1
        a = dqn.choose_action(s)
        # take action
        s_, r, done, info = env.step(a)

        s = s_
        
        # Print data
        if done:
           sum_reward += t
           result.append(t)
           print("Episode %d finished after %f time steps" % (episode, t))
           break
        if t == MAX_STEPS-1:
           sum_reward += t 
           result.append(t)
           print("Episode %d terminated after %f MAX steps" % (episode, t))
result = np.array(result)
print ("mean: %f ; std: %f" % (result.mean(),result.std()) )
plt.figure(1)
plt.plot(e_data,y_data)
plt.figure(2)
plt.plot(x_data,r_data)