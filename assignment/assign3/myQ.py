# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 17:13:44 2017

@author: Sharey
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 21:09:09 2017

@author: Sharey
"""
import gym
import numpy as np
import random
import math

one_degree = 0.0174532   # 2pi/360 */
MAX_STEP = 20000
NUM_TEST = 200

class MyQlearning():
    
    def __init__(self,gama = 0.99,alpha = 0.5):
        self.alpha = alpha
        self.gama  = gama
        
    def state(state):
        bucket_indice = []
        STATE_BOUNDS = self.STATE_BOUNDS
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
    
    def get_learning_rate(self,t):
        return max(0.1, min(0.5, 1.0 - math.log10((t+1)/25)))
    def get_explore_rate(self,t):
        return max(0.01, min(1, 1.0 - math.log10((t+1)/25)))
    
    
        
    def Qlearning(self,env,STATE_BOUNDS,NUM_BUCKETS):
        
        
        
        self.STATE_BOUNDS = STATE_BOUNDS
        self.gama = 0.99
        NUM_ACTIONS = env.action_space.n
        Q = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))
        
        def select_action(state,epsilon):
            # Select a random action
            if random.random() < epsilon:
                action = env.action_space.sample()
            # Select the action with the highest q
            else:
                action = np.argmax(Q[state])
            return action
        
        def state(state):
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
        
        for episode in range (1000):
            observation = env.reset()
            #env.render()
            self.alpha = self.get_learning_rate(episode)
            epsilon = self.get_explore_rate(episode)
            x = state(observation)
            for t in range(20000):

                a = select_action(x,epsilon)
                observation, reward, done, info = env.step(a)
                
                     
                #print (reward)
                #x_next = state_to_bucket(observation)
                x_next = state(observation)
                #a_next = self.polynomial(pi[x])
                a_next = np.argmax(Q[x_next])
                #reward = observation[0]
                Q[x][a] = Q[x][a] + self.alpha*(reward + \
                 self.gama*Q[x_next][a_next] - Q[x][a])
    
                x = x_next
                a = a_next
                
                if done:
                    print("Episode %d finished after %f  steps " % (episode, t))
                    break
                
        
        print ("-----------test---------------")
        sum_reward = 0
        NUM_TEST = 200
        MAX_STEP = 20000
        for episode in range(NUM_TEST):

            # Reset the environment
            observation = env.reset()
            x = state(observation)
            

            for t in range(MAX_STEP):
                
                #env.render()
    
                # Select an action
                action = np.argmax(Q[x])
                # Execute the action
                observation, reward, done, info = env.step(action)
                
    
                # Observe the result
                x = state(observation)
    
                # Setting up for the next iteration
    
                # Print data
                if done:
                   sum_reward += t 
                   #print("Episode %d finished after %f time steps" % (episode, t))
                   break
        print ("average:",sum_reward/NUM_TEST)    
        return 

if __name__ == '__main__':
    
    learner = MyQlearning()
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    NUM_BUCKETS = (1, 1, 6, 5)  # (x, x', theta, theta')
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))    
    STATE_BOUNDS[1] = [-0.5, 0.5]
    STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
    Q = learner.Qlearning(env,STATE_BOUNDS,NUM_BUCKETS)
    
    
    
