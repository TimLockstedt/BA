# Test, umlegung der Winkel der ODFs

import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import vispy_odf


from new_lib import *


# Amps = np.reshape(Amps, (5,5,5,phi.shape[0]))
phi, theta = fibonacci_sphere(1500)


alpha, beta = phi, theta

x = (np.cos(beta))
y = (np.sin(alpha)*np.sin(beta))
z = (-np.cos(alpha)*np.sin(beta))

theta_= np.arccos(z)
phi_ = np.arctan2(y,x)

x_ = np.cos(phi_) * np.sin(theta_)
y_ = np.sin(phi_) * np.sin(theta_)
z_ = np.cos(theta_)

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.set_title(f"alpha, beta")
ax.scatter(x,y,z)


fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.set_title(f"kugelkoord")
ax.scatter(x_,y_,z_)



################################################################
# fig = plt.figure()
# ax = fig.add_subplot(projection = "3d")
# x = Amps[2,1,3,:] * np.cos(phi) * np.sin(theta)
# y = Amps[2,1,3,:] * np.sin(phi) * np.sin(theta)
# z = Amps[2,1,3,:] * np.cos(theta)
# ax.scatter(x, y, z)

# fig = plt.figure()
# ax = fig.add_subplot(projection = "3d")
# x = Amps[1,2,3,:] * np.cos(phi) * np.sin(theta)
# y = Amps[1,2,3,:] * np.sin(phi) * np.sin(theta)
# z = Amps[1,2,3,:] * np.cos(theta)
# ax.scatter(x, y, z)



# fig = plt.figure()
# ax = fig.add_subplot(projection = "3d")
# x = Amps[2,1,3,:] * np.cos(phi) * np.sin(theta)
# y = Amps[2,1,3,:] * np.sin(phi) * np.sin(theta)
# z = Amps[2,1,3,:] * np.cos(theta)
# ax.scatter(x, y, z)
# x = Amps[1,2,3,:] * np.cos(phi) * np.sin(theta)
# y = Amps[1,2,3,:] * np.sin(phi) * np.sin(theta)
# z = Amps[1,2,3,:] * np.cos(theta)
# ax.scatter(x, y, z)
# plt.show()
################################################################
# for i in range(5):
#     x = Amps[i,2,3,:] * np.cos(phi) * np.sin(theta)
#     y = Amps[i,2,3,:] * np.sin(phi) * np.sin(theta)
#     z = Amps[i,2,3,:] * np.cos(theta)


#     fig = plt.figure()
#     ax = fig.add_subplot(projection = "3d")
#     ax.scatter(x, y, z)



# for i in range(5):
#     x = Amps[2,i,3,:] * np.cos(phi) * np.sin(theta)
#     y = Amps[2,i,3,:] * np.sin(phi) * np.sin(theta)
#     z = Amps[2,i,3,:] * np.cos(theta)


#     fig = plt.figure()
#     ax = fig.add_subplot(projection = "3d")
#     ax.scatter(x, y, z)
    

# plt.show()
################################################################
plt.show()