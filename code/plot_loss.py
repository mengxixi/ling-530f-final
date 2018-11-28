import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

log_fname = "log.20181127"

losses = []
with open(log_fname, 'r') as f:
    for line in f:
        if "running loss" in line:
            loss = line.split(' ')[-1]
            try:
                losses.append(float(loss))
            except:
                print(line)

plt.plot(losses)
plt.savefig("convergence.png")
