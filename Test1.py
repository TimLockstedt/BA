from Test_lib import *


# def koords_in_kegel_x(range_r = 4):
#     # Generiert die Indizes innerhalb eines Kegels entlang der x-Achse mit länge range_r 


#     return (x_mask, y_mask, z_mask)


def Kegel_Hybrid(x_koord = 0, y_koord = 0, z_koord = 0, range_r = 4, alpha = np.array([]), beta = np.array([]), no_dup=False):
    # Rotation zuerst um die y-Achse winkel Beta und anschließend um die x-Achse winkel Alpha

    x = np.linspace(-range_r*2, range_r*2, range_r*4+1)
    y = np.copy(x)
    z = np.copy(x)

    # Erstelle das Gitter für x, y und z
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)

    # Mask für innerhalb des Zylinders
    mask = ((z_grid**2 + y_grid**2) < x_grid**2) & (0 < x_grid) & (range_r > x_grid)
    # Filtere Punkte innerhalb des Kegels
    x_mask = x_grid[mask]
    y_mask = y_grid[mask]
    z_mask = z_grid[mask]

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    # Erzeuge 3D-Arrays für x, y und z Koordinaten
    rotation_matrix = np.empty((len(alpha), 3, 3))

    # Fülle die Rotationsmatrizen für alle Winkel gleichzeitig
    rotation_matrix[:, 0, 0] = cos_beta.flatten()
    rotation_matrix[:, 0, 1] = np.zeros_like(cos_beta).flatten()
    rotation_matrix[:, 0, 2] = sin_beta.flatten()

    rotation_matrix[:, 1, 0] = (sin_alpha*sin_beta).flatten()
    rotation_matrix[:, 1, 1] = cos_alpha.flatten()
    rotation_matrix[:, 1, 2] = (-sin_alpha*cos_beta).flatten()

    rotation_matrix[:, 2, 0] = (-cos_alpha*sin_beta).flatten()
    rotation_matrix[:, 2, 1] = sin_alpha.flatten()
    rotation_matrix[:, 2, 2] = (cos_alpha*cos_beta).flatten()
    

    # Dimension der x,y,z anheben
    x_mask_3d = np.tile(x_mask.ravel(), (len(alpha.T), 1))
    y_mask_3d = np.tile(y_mask.ravel(), (len(alpha.T), 1))
    z_mask_3d = np.tile(z_mask.ravel(), (len(alpha.T), 1))

    # Wende die Rotationsmatrix auf die x- und y-Koordinaten an
    result = np.dot(rotation_matrix, np.vstack([x_mask_3d, y_mask_3d, z_mask_3d]))
    # x = result[i,0,:]
    # y = result[i,1,:]
    # z = result[i,2,:]

    result[:,0,:] += x_koord
    result[:,1,:] += y_koord
    result[:,2,:] += z_koord

    # Falls gewollt werden hier die doppelten Punkte die durch das Runden auftreten herausgefilert und mit den Koordinaten 0,0,0 erstetzt
    return result


number_of_winkel = 1
rng = np.random.default_rng(random.randint(100000,10000000000))
beta = np.arccos(1-2*(rng.random(number_of_winkel).reshape(number_of_winkel,1)))
alpha = rng.random(number_of_winkel).reshape(number_of_winkel,1)*math.pi*2
range_r = 7
x_koord, y_koord, z_koord = 0,0,0

result = Kegel_Hybrid(x_koord, y_koord, z_koord, range_r, alpha, beta)

Ceil = np.ceil(result)
Floor = np.floor(result)
Round = np.round(result)

x_possible = [Ceil[:,0,:], Floor[:,0,:], Round[:,0,:]]
y_possible = [Ceil[:,1,:], Floor[:,1,:], Round[:,1,:]]
z_possible = [Ceil[:,2,:], Floor[:,2,:], Round[:,2,:]]


res = np.empty((*result.shape[:2], result.shape[2]*27))
print(res.shape)
counter = 0
for i in x_possible:
    for j in y_possible:
        for k in z_possible:
            res[:,:,counter*result.shape[2]:(counter+1)*result.shape[2]] = np.swapaxes(np.array([i,j,k]), 0,1)
            counter += 1


fig = plt.figure()
ax = fig.add_subplot(projection = "3d")

ax.scatter(*res[0,:,:])
# ax.scatter(*Test2[0,:,:])
# ax.scatter(*Test3[0,:,:])

plt.show()