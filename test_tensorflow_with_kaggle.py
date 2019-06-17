'''
Created on Jan 24, 2017

@author: aelsalla
'''

#Simple two layer neural net minimizing the mean squared value. I am trying to switch to R2 loss later (see my attempt in the code)
import kagglegym

import tensorflow as tf
print(tf.__version__)
import tensorflow.contrib.layers as layers
import tensorflow.contrib.losses as losses

# Create environment
env = kagglegym.make()

# Get first observation
observation = env.reset()


# Model
N_FEATURES=108
LEARNING_RATE = 0.001

x = tf.placeholder(tf.float32, shape=(None, N_FEATURES))
y = tf.placeholder(tf.float32, shape=(None,1))
p = tf.placeholder(tf.float32)
logits = layers.fully_connected(x, 200, activation_fn=tf.nn.relu)
logits = layers.dropout(logits, keep_prob=p)
logits = layers.fully_connected(x, 200, activation_fn=tf.nn.relu)
logits = layers.dropout(logits, keep_prob=p)
y_ = layers.fully_connected(logits, 1)
loss = losses.mean_squared_error(y, y_)

# Objective
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-y_)) / tf.square(y_ - tf.reduce_mean(y_))) # Equivalent to minimize R2

train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


# Data
mean_vals = observation.train.mean()
traindf = observation.train.drop(axis=1, labels=["id", "timestamp"]).fillna(mean_vals)

Y_train = traindf["y"]
X_train = traindf.drop(axis=1, labels=["y"])


# Training:
#__________

num_examples = X_train.shape[0]
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
        feeding_dict = { x: X_train.iloc[start:(start+batch_size)].values,
                        y: Y_train.iloc[start:(start+batch_size)].values.reshape(-1, 1),
                       p:0.5}
        start+=batch_size

        _, l  = sess.run([train_op, loss], feed_dict=feeding_dict)

        if not(batch_idx%1000):
            print("Loss on batch {}: {}".format(batch_idx, l))




# Integrate to kaggle

#import tensorflow.contrib.metrics as metrics

#smse, smse_update_op = metrics.streaming_mean_squared_error(y, y_)

sess.run(tf.initialize_local_variables())
rewards = []
n = 0
while True:
    target = observation.target
    features = observation.features.drop(axis=1, labels=["id", "timestamp"])
    feeding_dict = { x: features.values, p: 1. }
    y_.eval(feed_dict=feeding_dict)
    #sess.run(smse_update_op, feed_dict=feeding_dict)
    #sess.run(y_, feed_dict=feeding_dict)
    output = y_.eval(feed_dict=feeding_dict)
    target.loc[:, 'y'] = output
    # Fill in perfect actions
    #perfect_action = env.df[env.df["timestamp"] == observation.features["timestamp"][0]][["id", "y"]].reset_index(drop=True)
    #target.loc[:, 'y'] = perfect_action
    observation, reward, done, info = env.step(target)
    if done:
        break
    rewards.append(reward)
    n = n + 1
print(info)
print(n)
print(rewards[0:15])

