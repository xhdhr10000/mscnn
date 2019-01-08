import numpy as np
from matplotlib import pyplot as plt
import sys
print(sys.argv)

# load crowd map file
mp_name = sys.argv[1]
den_name = sys.argv[2]
mp = np.load(mp_name)
mp = np.squeeze(mp)

# crowd counts
people_counts = round(sum(sum(mp)))
print('Count: %d' % people_counts)

# show
plt.imsave('density/' + den_name + ".png", mp, cmap=plt.get_cmap('jet'))
#plt.imshow(mp, cmap=plt.get_cmap('jet'))
