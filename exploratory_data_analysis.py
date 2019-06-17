'''
Created on Jan 23, 2017

@author: aelsalla
'''

import kagglegym
from rope.base import resourceobserver


# Create environment
env = kagglegym.make()

# Get first observation
observation = env.reset()

'''
print(observation.train.dropna().shape)

for col in observation.train.columns:
    print(col)

print(len(env.unique_timestamp))

for time_stamp in env.unique_timestamp:
    print(time_stamp) # Ordered, no missing
    
print(len(observation.features))

'''
#print(observation.train.loc[1,:])
observation.train = observation.train.dropna()
print(len(observation.train))
'''
for index,row in observation.train.iterrows():
    #print(row)
    print(row.technical_42)
    #print(row["timestamp"])
'''    
    