# Test, umlegung der Winkel der ODFs

import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import vispy_odf




from new_lib import *

array = np.arange(0,1.001,0.001)
array1 = np.empty(array.shape[0]**2)
array2 = np.copy(array1)

count = 0
for a1 in array:
    for a2 in array:
        array1[count] = a1
        array2[count] = a2
        count += 1

# array1 = np.copy(array)
# array2 = np.copy(array)
# np.random.shuffle(array1)
# np.random.shuffle(array2)

theta = np.arccos(1-2*array1)
phi = 2*np.pi*array2

alpha = phi
beta = theta


x = (np.cos(beta))
y = (np.sin(alpha)*np.sin(beta))
z = (-np.cos(alpha)*np.sin(beta))


x_ = np.cos(phi) * np.sin(theta)
y_ = np.sin(phi) * np.sin(theta)
z_ = np.cos(theta)


fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.scatter(x, y, z)
ax.set_title(f"homogene Kugel?")

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.scatter(x_, y_, z_)
ax.set_title(f"homogene Kugel?")

plt.show()