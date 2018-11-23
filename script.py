import os
import random
random.seed(0)
data_dir = os.path.join(".","data","gigaword","train")
output = os.path.join(".","data","gigaword","train_sample.txt")
with open(output,'w+') as fo:                
        for fname in os.listdir(data_dir):
                fpath = os.path.join(data_dir, fname)
                with open(fpath) as f:
                        for line in f:
                                tmp = random.random()
                                if tmp < 0.4:
                                        continue
                                fo.write(line)