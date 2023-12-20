# Test, umlegung der Winkel der ODFs

import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import vispy_odf


from new_lib import *


# Amps = np.reshape(Amps, (5,5,5,phi.shape[0]))
phi, theta = np.array([0,0,0]),np.array([0,np.pi/2,np.pi/4])
result = koords_in_kegel(5, phi, theta)


alpha, beta = phi, theta
# x_ = np.cos(alpha) * np.sin(beta)
# y_ = np.sin(alpha) * np.sin(beta)
# z_ = np.cos(beta)




x = (np.cos(beta))
y = (np.sin(alpha)*np.sin(beta))
z = (-np.cos(alpha)*np.sin(beta))

theta_= np.arccos(z)
phi_ = np.arctan2(y,x)

x_ = np.cos(phi_) * np.sin(theta_)
y_ = np.sin(phi_) * np.sin(theta_)
z_ = np.cos(theta_)

origin = [0,0,0]
X, Y, Z = zip(origin,origin,origin) 
# U, V, W = zip(p0,p1,p2)

for i, res in enumerate(result):
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    ax.set_title(f"phi={phi[i]}, theta={theta[i]}")
    ax.scatter(*res)
    ax.quiver(X,Y,Z,x[i],y[i],z[i])
    ax.quiver(X,Y,Z,x_[i],y_[i],z_[i])
    ax.scatter(x[i], y[i], z[i])

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