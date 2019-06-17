'''
Created on Jan 23, 2017

@author: aelsalla
'''

import kagglegym

#kagglegym.test()

# Here's an example of loading the CSV using Pandas's built-in HDF5 support:

import pandas as pd

with pd.HDFStore("../../Data/train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")
    
print len(df)


# Create environment
env = kagglegym.make()

# Get first observation
observation = env.reset()

# Get length of the train dataframe
len(observation.train)

# Get number of unique timestamps in train
len(observation.train["timestamp"].unique())





# Note that this is half of all timestamps:
len(df["timestamp"].unique())

# Here's proof that it's the first half:
unique_times = list(observation.train["timestamp"].unique())
(min(unique_times), max(unique_times))

# Look at the first few rows of the features dataframe
observation.features.head()

# Look at the first few rows of the target dataframe
observation.target.head()


env = kagglegym.make()
observation = env.reset()

print(observation.train.dropna().shape)

for col in observation.train.columns:
    print(col)
observation.train["y"].hist(bins=100)
observation.train.dropna()["technical_5"].hist(bins=100)

 






n = 0
rewards = []
while True:
    target = observation.target
    target.loc[:, 'y'] = 0.01
    # Fill in perfect actions
    #perfect_action = env.df[env.df["timestamp"] == observation.features["timestamp"][0]][["id", "y"]].reset_index(drop=True)
    #target.loc[:, 'y'] = perfect_action
    observation, reward, done, info = env.step(target)
    if done:
        break
    rewards.append(reward)
    n = n + 1
    
