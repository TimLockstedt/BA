from Test_lib import *
factor = 1
# number_of_winkel = 50
# rng = np.random.default_rng(random.randint(100000,10000000000))
# beta = np.arccos(1-2*(rng.random(number_of_winkel).reshape(number_of_winkel,1)))
# alpha = rng.random(number_of_winkel).reshape(number_of_winkel,1)*math.pi*2
beta = np.array([1.1,1.1])
alpha = np.array([1.1,1.2])

cos_alpha = np.cos(alpha)
sin_alpha = np.sin(alpha)
cos_beta = np.cos(beta)
sin_beta = np.sin(beta)
# Karthesische Koordinate der symetrieachse berehnen
x = (cos_beta)
y = (sin_alpha*sin_beta)
z = (-cos_alpha*sin_beta)

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")



x1,y1,z1 = np.zeros((x.shape[0],3)).T

ax.quiver(x1,y1,z1,x*factor,y*factor,z*factor, arrow_length_ratio = 0.1)
ax.set_xlim(-2*factor,2*factor)
ax.set_ylim(-2*factor,2*factor)
ax.set_zlim(-2*factor,2*factor)
plt.show()