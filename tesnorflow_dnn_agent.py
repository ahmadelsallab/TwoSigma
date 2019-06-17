import kagglegym
import numpy as np
import pandas as pd
from DNNAgent import DNNAgent



def get_train_data(observation, col):
    

    
    # Note that the first observation we get has a "train" dataframe
    print("Train has {} rows".format(len(observation.train)))
    
    # The "target" dataframe is a template for what we need to predict:
    print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))
    
    
    traindf = observation.train.drop(axis=1, labels=["id", "timestamp"]).dropna()
    
    y = traindf["y"]
    x = traindf.drop(axis=1, labels=["y"])   
    
    return x, y

def get_test_data(observation, cols):
    features = observation.features.drop(axis=1, labels=["id", "timestamp"])
    #mean_vals = observation.features.mean()    
    #observation.features.fillna(mean_vals, inplace=True)
    return features.values


# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Columns for training
#cols = [col for col in train_data_cleaned.columns if "technical_" in col]
cols = 'technical_20'
#print(cols)
# Extract the train data out of the data frame
[x, y] = get_train_data(observation, cols)

N_FEATURES=108
LEARNING_RATE = 0.001
DNN = DNNAgent(N_FEATURES, LEARNING_RATE)

DNN.train(x, y)



while True:
    
    x_test = get_test_data(observation, cols)
    ypred = DNN.predict(x_test)
    observation.target.y = ypred

    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break