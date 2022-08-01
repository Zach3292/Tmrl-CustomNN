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

        if random.uniform(0, 1) < epsilon:
            action = np.array([1.0, 0, random.uniform([-1, 1])]) # Pick a new action for this state.
        else:
            action = np.array([1.0, 0, (np.argmax(q_table[state])) - 1]) # Pick the action which has previously given the highest reward.
        

        next_obs, rew, term, trun, info = env.step(action) 

        next_state = obsToState(next_obs)
        
        old_value = q_table[state, action] # Retrieve old value from the q-table.
        next_max = np.max(q_table[next_state])

        # Update q-value for current state.
        new_value = (1 - alpha) * old_value + alpha * (rew + gamma * next_max)
        q_table[state, action] = new_value

        if rew == -10: # Checks if agent attempted to do an illegal action.
            penalties += 1

        state = next_state
        
    if i % 100 == 0: # Output number of completed episodes every 100 episodes.
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
        obs, rew, term, trun, info = env.step(action) 

        if rew == -10:
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



