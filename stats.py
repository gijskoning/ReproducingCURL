import plot_data as help
import numpy as np

files = ["CURL_100k", "CURL_50k", "CURL_5k", "SAC-AE_100k", "SAC-AE_50k", "SAC-AE_5k", ]

for f in files:

    filename = f + "/train.log"

    time_steps, rewards, time_stamps, time_elapsed = help.read_file(filename, 62500)
    rewards = np.array(rewards)
    
    avg = np.mean(rewards[-21:])
    print(f, ":", avg)
