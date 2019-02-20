# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 23:36:59 2017

@author: Sharey
"""

import nltk
from nltk.corpus import brown
import collections
import numpy as np
import myHMM



# Estimating P(wi | ti) from corpus data using Maximum Likelihood Estimation (MLE):
# P(wi | ti) = count(wi, ti) / count(ti)
#  add an artificial "START" tag at the beginning of each sentence, and
#  add an artificial  "END"  tag at the end of each sentence.
def estimate_para():
    
    brown_tags_words = [ ]
    bts = []
    wordcounter = collections.Counter()
    
    tagcounter  = collections.Counter()
    
    
    for sent in brown.tagged_sents(tagset = "universal"):
        bts.append(sent)
        # sent is a list of word/tag pairs
        # add START/START at the beginning
        brown_tags_words.append( ("START", "START") )
        # change the order of tag and word
        brown_tags_words.extend([ (tag, word) for (word, tag) in sent ])
        # add END/END at the end
        brown_tags_words.append( ("END", "END") )
    
    cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
    # conditional probability distribution
    cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)
    
    # Estimating P(ti | t{i-1}) from corpus data using Maximum Likelihood Estimation (MLE):
    # P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
    brown_tags = [tag for (tag, word) in brown_tags_words ]
    
    # make conditional frequency distribution:
    # count(t{i-1} ti)
    cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
    # make conditional probability distribution, using
    # maximum likelihood estimate:
    # P(ti | t{i-1})
    cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)
    
    for (tag , word) in brown_tags_words:
        tagcounter[tag] += 1
        wordcounter[word] += 1
        
    tag2list = tagcounter.most_common()
    word2list = wordcounter.most_common()
    
    tag2index = {x[0]: i for i, x in enumerate(tag2list)}
    word2index = {x[0]: i for i, x in enumerate(word2list)}
    
    index2tag = {v:k for k, v in tag2index.items()}
    index2word = {v:k for k, v in word2index.items()}
    
    
    
    N = len(index2tag)   #状态空间
    M = len(index2word)  #观测空间
    A = np.zeros((N,N))  #模型参数A
    B = np.zeros((N,M))  #模型参数B
    pi = np.zeros(N)     #模型参数pi
    
    for i in range (N):
        for j in range (N):
            A[i][j] = cpd_tags[index2tag[i]].prob(index2tag[j])
                
    
    for i in range (N):
        for j in range (M):
            B[i][j] = cpd_tagwords[index2tag[i]].prob(index2word[j])
    
    
    
    pi[tag2index["START"]] = 1
    
    return A,B,pi,word2index,index2tag,brown_tags_words


#把输入的句子转换为观测序列O；如果输入错误则报错
def generate_o(sentence,word2index): 
    
    sentence = sentence.split()
    sentence.insert(0,"START")
    sentence.append("END")
    
    for word in sentence:
        if word2index.get(word) ==None:
            print("error:some word not in the dictionary")
            return [],False
  
    sent = [word2index[x] for x in sentence]    
    o = np.array(sent)
    return o,True

#调用myHMM在中的Viterbi算法
def postag_sentence(A,B,o,pi,index2tag):
    hmm = myHMM.myHMM()
    path = hmm.HMMViterbi(A,B,o,pi)    #调用维特比算法
    tags = [index2tag[x] for x in path]   
    return tags

if __name__ == '__main__':
    
    A,B,pi,word2index,index2tag,brown_tags_words = estimate_para() #初始化模型参数
    sentence = input ("please input the sentece(E to exit):") #获取用户输入；输入'E'退出
    while (sentence !="E"):
        
        o,condition = generate_o(sentence,word2index)
        if condition:
            tags = postag_sentence(A,B,o,pi,index2tag)
            print (tags)        
               
        sentence = input ("please input the sentece(E to exit):")
    print ("***program end***")
    