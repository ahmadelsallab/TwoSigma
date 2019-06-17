'''
Created on Feb 9, 2017

@author: aelsalla
'''

import numpy as np
import tensorflow as tf
print(tf.__version__)
import tensorflow.contrib.layers as layers
import tensorflow.contrib.losses as losses

class DNNAgent(object):
    '''
    classdocs
    '''


    def __init__(self, N_FEATURES, LEARNING_RATE):
        '''
        Constructor
        '''
        self.x = tf.placeholder(tf.float32, shape=(None, N_FEATURES))
        y = tf.placeholder(tf.float32, shape=(None,1))
        self.p = tf.placeholder(tf.float32)
        self.logits = layers.fully_connected(self.x, 56, activation_fn=tf.nn.relu)
        self.logits = layers.dropout(self.logits, keep_prob=self.p)
        self.logits = layers.fully_connected(self.x, 56, activation_fn=tf.nn.relu)
        self.logits = layers.dropout(self.logits, keep_prob=self.p)
        y_ = layers.fully_connected(self.logits, 1)
        self.loss = losses.mean_squared_error(y, y_)
        
        # Objective
        # loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-y_)) / tf.square(y_ - tf.reduce_mean(y_))) # Equivalent to minimize R2
        
        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)        
    
    def train(self, x_train, y_train):
        num_examples = x_train.shape[0]
        batch_size = 32
        n_epoch = 2
        n_batch = int(num_examples / batch_size)
        print("Feeding {} batches per epoch".format(n_batch))
        start = 0
        
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        
        for _ in range(n_epoch):
            start = 0
            for batch_idx in range(n_batch-1):
            #for batch_idx in range(15):
                feeding_dict = { x: x_train.iloc[start:(start+batch_size)].values,
                                 y: y_train.iloc[start:(start+batch_size)].values.reshape(-1, 1),
                                 p:0.5}
                start+=batch_size
        
                _, l  = sess.run([self.train_op, self.loss], feed_dict=feeding_dict)
        
                if not(batch_idx%1000):
                    print("Loss on batch {}: {}".format(batch_idx, l))
                    
    def predict(self, x):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_local_variables())
        feeding_dict = { self.x: x, self.p: 1. }
        self.y_.eval(feed_dict=feeding_dict)
        #sess.run(smse_update_op, feed_dict=feeding_dict)
        #sess.run(y_, feed_dict=feeding_dict)
        output = self.y_.eval(feed_dict=feeding_dict)        
        return output
    
    def rl_update(self, reward, state, action):
        return    