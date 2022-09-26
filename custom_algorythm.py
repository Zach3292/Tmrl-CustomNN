from tmrl import get_environment
from time import sleep
from os import system, name
import numpy as np
import random
import os

# local imports
from tmrl.custom.utils.window import WindowInterface
from tmrl.custom.utils.tools import Lidar

data_path = "C:\\Users\\Zach\\Documents\\001 TMRL\\nparray"
created_folder = "C:\\Users\\Zach\\Documents\\001 TMRL\\nparray"


def clear(): 
  
    # Clear on Windows.
    if name == 'nt': 
        _ = system('cls')
  
    # Clear on Mac and Linux. (os.name is 'posix') 
    else: 
        _ = system('clear')

env = get_environment()  # retrieve the TMRL Gym environment

window_interface = WindowInterface("Trackmania")
lidar = Lidar(window_interface.screenshot())

# row = different state aka array of 4 lidar value, columns = different action aka [gas, break, steer], analog between -1.0 and +1.0

# the 20001 state are representing a sterring value from 0 to 2 with 4 decimal point and the 3 simple action for straight, right and left

if os.path.isfile("C:\\Users\\Zach\\Documents\\001 TMRL\\nparray\\neuralNetwork.npy"):
    q_table = np.load("C:\\Users\\Zach\\Documents\\001 TMRL\\nparray\\neuralNetwork.npy")
else:
    q_table = np.zeros([201, 201])



training_episodes = 20000 # Amount of times to run environment while training.
display_episodes = 10 # Amount of times to run environment after training.

# Hyperparameters
alpha = 0.1 # Learning Rate
gamma = 0.6 # Discount Rate
epsilon = 0.25 # Chance of selecting a random action instead of maximising reward.

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
    steer = pow(steer, 3)
    state = round(steer, 3) + 1  # To transform the deviation into the 201 state of the array
    state *= 100
    state = int(state)
    return state

"""Training the Agent"""



sleep(1.0)

for i in range(training_episodes):
    obs = env.reset()
    state = obsToState(obs)
    term = False
    penalties, rew, = 0, 0
    
    while not term:
        
        img = window_interface.screenshot()[:, :, :3] # Display Live Lidar on screen
        lidar.lidar_20(img, True)

        if random.uniform(0, 1) < epsilon:
            action = round(random.uniform(0, 200)) # Pick a new action for this state.
        else:
            action = np.argmax(q_table[state]) # Pick the action which has previously given the highest reward.
        

        next_obs, rew, term, info = env.step(np.array([1.0, 0, (action/100)-1])) 

        next_state = obsToState(next_obs)
        
        old_value = q_table[state, action] # Retrieve old value from the q-table.
        next_max = np.max(q_table[next_state])

        if next_obs[0] < 20:
            rew -= 10

        if next_obs[0] > 200:
            rew += 10

        # Update q-value for current state.
        new_value = (1 - alpha) * old_value + alpha * (rew + gamma * next_max)
        q_table[state, action] = new_value

        path = os.path.join(data_path, created_folder, 'neuralNetwork')
        with open('{}.npy'.format(path), 'wb') as f:
            np.save(f, q_table)

        if next_obs[0] < 20:
            penalties += 1

        if rew == 0: # Checks if agent attempted to do an illegal action or bad action.
            penalties += 1

        state = next_state

        clear()
        print(f"Episode: {i}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {rew}")



print("Training finished.\n")


total_epochs, total_penalties = 0, 0

for _ in range(display_episodes):
    obs = env.reset()
    epochs, penalties, rew = 0, 0, 0
    
    term = False
    
    while not term:

        state = obsToState(obs)

        action = np.argmax(q_table[state])
        obs, rew, term, info = env.step(np.array([1.0, 0, (action/100)-1])) 

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



