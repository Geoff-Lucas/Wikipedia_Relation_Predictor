#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:26:25 2019

@author: geoffreylucas
"""

import tensorflow as tf
import random
import pickle


class Critic:
    
    def __init__(self, n_features, learning_rate, n_hidden1, n_hidden2):
        """
        Critic initialization
        :param n_features: int, number of valid actions for a given environment
        :param learning_rate: float, learning rate for agent
        :param n_hidden1: int, number of units for hidden layer1
        :param n_hidden2: int, number of units for hidden layer2
        """
        
        self.startup(n_features, learning_rate, n_hidden1, n_hidden2)        
        
    
    def startup(self, n_features, learning_rate, n_hidden1, n_hidden2):
        """
        Critic startup
        :param n_features: int, number of valid actions for a given environment
        :param learning_rate: float, learning rate for agent
        :param n_hidden1: int, number of units for hidden layer1
        :param n_hidden2: int, number of units for hidden layer2
        """
        
        # Placeholders
        self.observation = tf.placeholder(tf.float32, shape=(None, n_features))
        self.reward = tf.placeholder(tf.float32, shape=(None, 1))
        
        # Layers Construction
        self.layer1 = tf.layers.dense(inputs = self.observation, 
                                      units = n_hidden1, 
                                      activation = tf.nn.relu, 
                                      use_bias = True,
                                      kernel_initializer = tf.initializers.glorot_normal,
                                      name = 'CriticHidden1')
        self.layer2 = tf.layers.dense(inputs = self.layer1, 
                                      units = n_hidden2, 
                                      activation = tf.nn.relu, 
                                      use_bias = True,
                                      kernel_initializer = tf.initializers.glorot_normal,
                                      name = 'CriticHidden2')
        self.layer3 = tf.layers.dense(inputs = self.layer2, 
                                      units = 1,
                                      name = 'ValueEstimate')
            
        # Optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate)
        self.loss = tf.reduce_mean(tf.square(self.reward - self.layer3))
        self.train = self.opt.minimize(self.loss)
        
'''        
Note: Referencing this website as a guide for a better design for the actor
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/
blob/master/contents/8_Actor_Critic_Advantage/AC_CartPole.py

- I was having problems gettin my actor to train, then finally realized my main
problem was I had the name of the train layer wrong and was never 
running it... DOH.
'''

class Actor:
    
    def __init__(self, n_features, n_output, learning_rate, n_hidden1, n_hidden2, scope):
        """
        Critic initialization
        :param n_features: int, number of valid actions for a given environment
        :param learning_rate: float, learning rate for agent
        :param n_hidden1: int, number of units for hidden layer1
        :param n_hidden2: int, number of units for hidden layer2
        """
        
        self.startup(n_features, n_output, learning_rate, n_hidden1, n_hidden2, scope)        
        
    
    def startup(self, n_features, n_output, learning_rate, n_hidden1, n_hidden2, scope):
        """
        Critic startup
        :param n_features: int, number of valid actions for a given environment
        :param learning_rate: float, learning rate for agent
        :param n_hidden1: int, number of units for hidden layer1
        :param n_hidden2: int, number of units for hidden layer2
        """
        with tf.variable_scope(scope):
            # Placeholders
            self.observation = tf.placeholder(tf.float32, shape = (1, n_features))
            self.td_error = tf.placeholder(tf.float32, shape = None)
            self.relation = tf.placeholder(tf.int32, shape = None)
            
            # Layers
            self.layer1 = tf.layers.dense(inputs = self.observation, 
                                          units = n_hidden1, 
                                          activation = tf.nn.relu, 
                                          use_bias = True,
                                          kernel_initializer = tf.random_normal_initializer(0.0, 0.1),
                                          name = 'ActorHidden1')
            self.layer2 = tf.layers.dense(inputs = self.layer1, 
                                          units = n_hidden2, 
                                          activation = tf.nn.relu, 
                                          use_bias = True,
                                          kernel_initializer = tf.random_normal_initializer(0.0, 0.1),
                                          name = 'ActorHidden2')
            self.layer3 = tf.layers.dense(inputs = self.layer2, 
                                          units = n_output, 
                                          activation = tf.nn.softmax, 
                                          use_bias = True,
                                          kernel_initializer = tf.initializers.glorot_normal,
                                          name = 'RelationProbability')
            
            self.log_probability = tf.log(self.layer3[0, self.relation])
            self.exp_v = tf.reduce_mean(self.log_probability * self.td_error)
            self.log_probability2 = tf.log(self.layer3)

            self.opt = tf.train.AdamOptimizer(learning_rate)
            self.loss = tf.reduce_mean(-1 * self.exp_v)
            self.train = self.opt.minimize(tf.square(-self.exp_v))
            
            
        
class ReplayBuffer:
    """ Class for Experience Replay"""
    
    
    def __init__(self):
        '''
        class initialization
        :self.relations: list to hold stored transitions
        '''        
        
        # For Collecting resulting transitions
        self.relations = []

    def sample(self, batch_size):
        
        temp = random.sample(self.relations, batch_size)   
        
        return temp
    
    def save_buffer(self):
        """
        Save current buffer contents to file
        """        
        
        # Code for saving to pickle
        pickle.dump(self.relations, open("buffer_contents.pkl", "wb"))
        print("Buffer saved\n")

    def load_buffer(self):
        """
        Load file of buffer contents
        """

        # To build the original list again.
        self.relations = pickle.load(open("buffer_contents.pkl", "rb"))
        print("Buffer laoded\n")
