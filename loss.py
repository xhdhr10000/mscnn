import numpy as np
import sys

ls = []
with open('eval.log') as f:
    lines = f.read().splitlines()
    for line in lines:
        if line == lines[-1]: break
        index = line.index('loss_value') + 12
        loss = float(line[index:].split(' ')[0])
        ls.append(loss)

print('Average loss: %f' % np.average(ls))
