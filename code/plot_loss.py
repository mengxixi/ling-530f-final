import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt



losses = []
with open("log.20181125", 'r') as f:
    for line in f:
        if "running loss" in line:
            loss = line.split(' ')[-1]
            losses.append(float(loss))

plt.plot(losses)
plt.savefig("convergence.png")
