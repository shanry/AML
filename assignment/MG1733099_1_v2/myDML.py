# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 24:00:00 2017

@author: Zhiyu
"""
# import your module here
import numpy as np
import time
from threading import Thread
import functools

# (global) variable definition here
TRAINING_TIME_LIMIT = 60*10

# class definition here

# function definition here
def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

@timeout(TRAINING_TIME_LIMIT)
def train(traindata):
    # 在此处完成你的训练函数，注意训练时间不要超过TRAINING_TIME_LIMIT(秒)。
    train_X=np.array(traindata[0])
    train_Y=np.array(traindata[1])
    global A
    A=np.eye(train_X.shape[1]) #object matrix
#    p=np.eye(train_X.shape[0]) #vote matrix 
    
    
    for epoc in range (10):
        
#        rand_sample=np.random.randint(0,train_X.shape[0],10)
#        pi=np.zeros(rand_sample.size)
#        F=np.zeros((train_X.shape[0])) # all votes
        for i in range (train_X.shape[0]):
#            i=rand_sample[ii]
#            i=np.random.randint(0,train_X.shape[0])
            index=(train_Y==train_Y[i])
            Xij=train_X[i,:]-train_X
            Ci=Xij[index]
            Ax=np.dot(A,Xij.transpose())
            f=(Ax*Ax).sum(axis=0)            
            f=np.exp(-f)
            f[i]=0
            p=f/f.sum()#softmax函数           
            
            #计算导数
            right=np.dot(Ci.transpose()*p[index],Ci)
            left=p[index].sum()*np.dot(Xij.transpose()*p,Xij)
            derivative=2*np.dot(A,(left-right))
            
            A=0.005*derivative+A
        #print(p[index].sum())   
#        if ((epoc%20)==0):
    print(p[index].sum())
    
        
    return 0

def Euclidean_distance(inst_a, inst_b):
    return np.linalg.norm(inst_a - inst_b)

def distance(inst_a, inst_b):
    return np.linalg.norm(np.dot(A,inst_a) - np.dot(A,inst_b))
    

# main program here
if  __name__ == '__main__':
    pass