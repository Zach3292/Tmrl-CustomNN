from ast import If
from tmrl import get_environment
from time import sleep
from os import system, name
import numpy as np
import random

from torch import round_

def clear(): 
  
    # Clear on Windows.
    if name == 'nt': 
        _ = system('cls')
  
    # Clear on Mac and Linux. (os.name is 'posix') 
    else: 
        _ = system('clear')

env = get_environment()  # retrieve the TMRL Gym environment

# row = different state aka array of 4 lidar value, columns = different action aka [gas, break, steer], analog between -1.0 and +1.0

# for now we'll only try to make the car steer so 3 simple state with lidar to see if the road is straight ahead, right or left and 3 simple action for straight, right and left
q_table = np.zeros([3, 3])
# 0 = left
# 1 = nothing
# 2 = right

training_episodes = 20000 # Amount of times to run environment while training.
display_episodes = 10 # Amount of times to run environment after training.

# Hyperparameters
alpha = 0.1 # Learning Rate
gamma = 0.6 # Discount Rate
epsilon = 0.1 # Chance of selecting a random action instead of maximising reward.

# For plotting metrics
all_epochs = []
all_penalties = []



# default observations are of shape: ((1,), (4, 19), (3,), (3,))
# representing: (speed, 4 last LIDARs, 2 previous actions)
# actions are [gas, break, steer], analog between -1.0 and +1.0


def obsToState(obs):
    deviation = obs[1].mean(0)
    deviation /= (deviation.sum() + 0.001)
    steer = 0
    for i in range(19):
        steer += (i - 9) * deviation[i]
    steer = - np.tanh(steer * 4)
    steer = min(max(steer, -1.0), 1.0)

    state = round(steer) + 1  # To transform the deviation into the 3 state of the array
    return state

"""Training the Agent"""

sleep(1.0)

for i in range(training_episodes):
    obs = env.reset()
    term = False
    penalties, rew, = 0, 0
    
    while not term:

        state = obsToState(obs)

        print(state)       

        # -------------BIG ISSUE!!----------------- 
        # State is always 1 no matter what the position of the car is
        #
        # When looking at the output of the steer variable in the obsToState function, 1/2 of the time, the 
        # value is 0.0044 making it round to zero + 1 meaning the state is 1/2 of the time equal to 1 in the  
        # function but when looking at the state here, it is always one, this 1/2 0.0044 value of the steer 
        # variable can be tracked all the way up to the deviation array that contain 1/2 the time exactly the 
        # same array value for some reason I don't know
        #
        # We need a way to only use the good value and discard the other one while allowing the training to 
        # not stop when we discard said value
        #------------------------------------------


        if random.uniform(0, 1) < epsilon:
            action = round(random.uniform(0, 2)) # Pick a new action for this state.
        else:
            action = np.argmax(q_table[state]) # Pick the action which has previously given the highest reward.
        

        next_obs, rew, term, info = env.step(np.array([1.0, 0, (action -1)])) 

        next_state = obsToState(next_obs)
        
        old_value = q_table[state, action] # Retrieve old value from the q-table.
        next_max = np.max(q_table[next_state])

        # Update q-value for current state.
        new_value = (1 - alpha) * old_value + alpha * (rew + gamma * next_max)
        q_table[state, action] = new_value

        if rew == 0: # Checks if agent attempted to do an illegal action or bad action.
            penalties += 1

        state = next_state
        
    if i % 1 == 0: # Output number of completed episodes every 100 episodes.
        print(f"Episode: {i}")

print("Training finished.\n")


total_epochs, total_penalties = 0, 0

for _ in range(display_episodes):
    obs = env.reset()
    epochs, penalties, rew = 0, 0, 0
    
    term = False
    
    while not term:

        state = obsToState(obs)

        action = np.argmax(q_table[state])
        obs, rew, term, info = env.step(np.array([1.0, 0, (action -1)])) 

        if rew == 0:
            penalties += 1

        epochs += 1
        clear()
        env.render()
        print(f"Timestep: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {rew}")
        sleep(0.15) # Sleep so the user can see the 

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {display_episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / display_episodes}")
print(f"Average penalties per episode: {total_penalties / display_episodes}")



