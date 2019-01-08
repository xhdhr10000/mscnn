import numpy as np
import matplotlib.pyplot as plt

ys = []
with open('train.log') as f:
    lines = f.read().splitlines()
    for line in lines:
        index = line.index('avg_loss') + 9
        loss = float(line[index:].split(' ')[0])
        ys.append(loss)

print('Average loss: %f' % np.average(ys))

xs = [i for i in range(len(ys))]
plt.title('loss')
plt.plot(xs, ys)
plt.show()
