import os
import numpy as np

data_path = "C:\\Users\\Zach\\Documents\\001 TMRL\\nparray"
created_folder = "C:\\Users\\Zach\\Documents\\001 TMRL\\nparray"

path = os.path.join(data_path, created_folder, 'testone')
with open('{}.npy'.format(path), 'wb') as f:
     np.save(f, np.arange(10))

array = np.load("C:\\Users\\Zach\\Documents\\001 TMRL\\nparray\\testone.npy")

print(array)

path = os.path.join(data_path, created_folder, 'testone')
with open('{}.npy'.format(path), 'wb') as f:
     np.save(f, np.arange(5))

array = np.load("C:\\Users\\Zach\\Documents\\001 TMRL\\nparray\\testone.npy")

print(array)