import numpy as np
import matplotlib.pyplot as plt


def build_random_walk(self):

    self.walk_list = [99] # initialize the random walk

    noise_list = []
    for i in range(1900):
        noise = -1 if np.random.random() < 0.5 else 1 # Create random noise
        noise_list.append(noise)
        self.walk_list.append(self.walk_list[-1] + noise)
    plt.plot(self.walk_list)
    plt.savefig(self.output_path+"randomwalk.png")
