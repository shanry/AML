# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 21:09:09 2017

@author: Sharey
"""
import gym
import numpy as np
import random
import math
v = []
rv = []
MAX1 = 3
MAX2 = 19
MAX3 = 15
MAX4 = 10
MAX5 = 5

one_degree = 0.0174532   # 2pi/360 */
env = gym.make('MountainCar-v0')

## Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (10,10 )  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = env.action_space.n # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
#STATE_BOUNDS[1] = [-0.5, 0.5]
#STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

class MyQlearning():
    
    def __init__(self,gama = 0.99,alpha = 0.5):
        self.alpha = alpha
        self.gama  = gama
        
    def echo(self):
        print (self.alpha,self.gama)
    def polynomial(self,p):
        #print (np.sum(p))
        #if not np.sum(p)== 1:
            #print (p,np.sum(p))
        assert np.sum(p)< 1.01
        assert len(p) > 1
        randvar = np.random.rand()
        sum = 0
        #print (len(p))
        #print (randvar)
        for i in range (len(p)):
            #print (i)
            sum += p[i] 
            if randvar <= sum:
                return i
        
    
    def state(self,observation):
        i = (observation[0]+1.2) // 0.1 
        j = (observation[1]+0.07) // 0.01
        
        return int (i+j*10)
    def box(self,observation):
        j = k = 0
        if observation[2] < -24*one_degree:
            j = 0
        elif observation[2] < -12*one_degree:
            j = 1
        elif observation[2] < 0:
            j = 2
        elif observation[2] < 12*one_degree:
            j = 3
        elif observation[2] < 24*one_degree:
            j = 4
        else:
            j = 5
        if observation[3] < -50*one_degree:
            k = 0
        elif observation[3] < 50*one_degree:
            k = 1
        else:
            k =2
        return j+k*6
    def state2(self,observation):
        i = (observation[0]+1.2) // 0.06
        j = (observation[1]+0.07) // 0.007
        
        if i >= MAX1:
            i = MAX1-1
        if j >= MAX2:
            j = MAX2-1
        return int (i*MAX1+j)
    def get_learning_rate(self,t):
        return max(0.5, min(0.5, 1.0 - math.log10((t+1)/500)))
    def get_explore_rate(self,t):
        return max(0.1, min(1, 1.0 - math.log10((t+1)/500)))
    
    def Qlearning(self,env):
        
        
        #v.append(observation[2])
        ACTS = env.action_space.n
        Q = np.zeros((MAX2*MAX3,ACTS))
        pi = np.ones((MAX2*MAX3,ACTS))
        pi = pi / ACTS
        #epsilon = 0.5
        #pi_greedy = pi*(1-epsilon)+epsilon/ACTS
        #print (pi_greedy.sum(axis = 1))
        def select_action(state,epsilon):
            # Select a random action
            if random.random() < epsilon:
                action = env.action_space.sample()
            # Select the action with the highest q
            else:
                action = np.argmax(Q[state])
            return action
        
        #for i in range (1000):
        for episode in range (20000):
            #print ("range",t)
            observation = env.reset()
            #env.render()
            self.alpha = self.get_learning_rate(episode)
            self.gama = 0.99
            epsilon = self.get_explore_rate(episode)
            x = self.state(observation)
            #x = state_to_bucket(observation)
            sum_reward = 0
            max_height = 0
            for t in range(250):
                #a = self.polynomial(pi_greedy[x])
                #a = np.random.binomial(1,pi_greedy[x][1])
                #env.render()
                
                a = select_action(x,epsilon)
                observation, reward, done, info = env.step(a)
                sum_reward += reward
                height = observation[0]
                if height > max_height:
                    reward += max_height -height
                    max_height = height
                     
                #print (reward)
                #x_next = state_to_bucket(observation)
                x_next = self.state(observation)
                #a_next = self.polynomial(pi[x])
                a_next = np.argmax(Q[x_next])
                #reward = observation[0]
                Q[x][a] = Q[x][a] + self.alpha*(reward + \
                 self.gama*Q[x_next][a_next] - Q[x][a])
                update = np.argmax(Q[x])
                pi[x] = 0
                pi[x][update] = 1                
                #pi_greedy = pi*(1-epsilon)+epsilon/ACTS
                x = x_next
                a = a_next
                
                if done:
                    print("Episode %d finished after %f  steps with reward %d" % (episode, t, sum_reward))
                    break
            
        return Q
    
if __name__ == '__main__':
    
    learner = MyQlearning()
    #learner.echo()
    env = gym.make('MountainCar-v0')
    Q = learner.Qlearning(env)
    
    
    