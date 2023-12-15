from sphere_lib import *

Amps = np.load("Amps.npy")
phi = np.load("phi.npy")
theta = np.load("theta.npy")
AODF_Amplitude = Amps[11]
# Scatterplot generieren
x = AODF_Amplitude[:,None] * np.cos(phi) * np.sin(theta)
y = AODF_Amplitude[:,None] * np.sin(phi) * np.sin(theta)
z = AODF_Amplitude[:,None] * np.cos(theta)
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
n = 10000
ax.scatter(x[:n],y[:n],z[:n])
plt.show()

mask = (y > -0.25) & (y < 0.25)
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
n = 10000
ax.scatter(x[mask],y[mask],z[mask])
plt.show()