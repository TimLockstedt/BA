from new_lib import *
range_r = 5
bands = 10
number_of_winkel = 250
sigma = 0.3


result, basis = get_result_basis(range_r, bands, number_of_winkel)

field_theta, field_phi = genereate_divergence()
ODFs = odf3.compute(field_theta[:,:,:,None], field_phi[:,:,:,None], np.ones(field_phi[:,:,:,None].shape), bands)[21-range_r-3:20+range_r+3,21-range_r-3:20+range_r+3,21-range_r-3:20+range_r+3,:]

limit_x, limit_y, limit_z = ODFs.shape[0],ODFs.shape[1],ODFs.shape[2] #20,20,20# 

result_rot = reverse_rotate_and_translate_data_noTranslation(result, number_of_winkel)
weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], 0, 0, sigma)


phi, theta = sphere(number_of_winkel)
factor_amp = 10
AODFs = np.array([
    Get_AODF_noRand_noCache_amp(ODFs,result, basis, weights,phi,theta,i,j,k, sigma=sigma, factor_amp=factor_amp, bands=bands)[0]
    for i in tqdm(range(range_r, limit_x - range_r))
    for j in range(range_r, limit_y - range_r)
    for k in range(range_r, limit_z - range_r)
])

Amps = np.array([
    Get_AODF_noRand_noCache_amp(ODFs,result, basis, weights,phi,theta,i,j,k, sigma=sigma, factor_amp=factor_amp, bands=bands)[1]
    for i in tqdm(range(range_r, limit_x - range_r))
    for j in range(range_r, limit_y - range_r)
    for k in range(range_r, limit_z - range_r)
])
length = int(np.round(AODFs.shape[0]**(1/3)))
AODFs = np.reshape(AODFs,(length,length,length,odf.get_num_coeff(bands)))
np.save(f"AODF_Kugel_Stern_NoRand_nocache_b{bands}", AODFs)
np.save(f"Amps_Kugel_Stern_NoRand_nocache_b{bands}", Amps)
odf.visualize_odf(AODFs[1,1,1,:],120,120)
plt.show()


# # phi, theta = fibonacci_sphere(750)
# # phi_, theta_ = fibonacci_sphere_2(750)

# # x = np.cos(phi) * np.sin(theta)
# # y = np.sin(phi) * np.sin(theta)
# # z = np.cos(theta)

# # x_ = np.cos(phi_) * np.sin(theta_)
# # y_ = np.sin(phi_) * np.sin(theta_)
# # z_ = np.cos(theta_)


# # fig = plt.figure()
# # ax = fig.add_subplot(projection = "3d")
# # ax.scatter(x, y, z)
# # ax.scatter(x_, y_,z_)
# # plt.show()



# import numpy as np
# from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
# import matplotlib.pyplot as plt

# def fibonacci_sphere(samples=1000):
#     points = []
#     phi = np.pi * (3. - np.sqrt(5.))  # Goldener Winkel

#     for i in range(samples):
#         y = 1 - (i / float(samples - 1)) * 2  # -1 to 1
#         radius = np.sqrt(1 - y * y)
#         theta = phi * i

#         x = np.cos(theta) * radius
#         z = np.sin(theta) * radius

#         points.append([x, y, z])

#     return np.array(points)

# def plot_spherical_points(points):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2])
#     plt.show()

# def main(num_points):
#     # Generiere Punkte auf der Einheitskugel mit gleichgroßen Dreiecken
#     points = fibonacci_sphere(num_points)

#     # Delaunay-Triangulation
#     tri = Delaunay(points)

#     # Voronoi-Diagramm
#     vor = Voronoi(points)

#     # Plotte die Punkte, Delaunay-Triangulation und Voronoi-Diagramm
#     plot_spherical_points(points)
#     plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
#     voronoi_plot_2d(vor)
#     plt.show()

#     # Gib die Winkel der äquidistanten Punkte aus
#     alpha = np.arctan2(points[:, 1], points[:, 0])
#     print("Winkel der äquidistanten Punkte:")
#     print(angles)

# if __name__ == "__main__":
#     num_points = 1000  # Anzahl der äquidistanten Punkte
#     main(num_points)

# phi, theta = fibonacci_sphere_2(1500)

# x = np.cos(phi) * np.sin(theta)
# y = np.sin(phi) * np.sin(theta)
# z = np.cos(theta)


# fig = plt.figure()
# ax = fig.add_subplot(projection = "3d")
# ax.scatter(x, y, z)
# plt.show()

# points = np.array([x, y, z]).T

# # Delaunay-Triangulation
# from scipy.spatial import Delaunay, Voronoi
# tri = Delaunay(points)

# # Berechne die äquidistanten Winkel in Kugelkoordinaten
# phi = []
# theta = []

# for simplex in tri.simplices:
#     for i in range(3):
#         p = points[simplex[i]]
#         r, azimuth, inclination = np.linalg.norm(p), np.arctan2(p[1], p[0]), np.arccos(p[2] / np.linalg.norm(p))

#         # Füge die Winkel zu den Listen hinzu
#         phi.append(azimuth)
#         theta.append(inclination)



# x = np.cos(phi) * np.sin(theta)
# y = np.sin(phi) * np.sin(theta)
# z = np.cos(theta)


# fig = plt.figure()
# ax = fig.add_subplot(projection = "3d")
# ax.scatter(x, y, z)
# plt.show()


# points = np.array([x, y, z]).T

# vor = Voronoi(points)

# # Verbessere die Punkte auf der Einheitskugel
# new_points = []

# for region in vor.regions:
#     if not -1 in region and len(region) > 0:
#         # Berechne den Schwerpunkt der Region
#         centroid = np.mean(np.array([vor.vertices[i] for i in region]), axis=0)
#         # Normalisiere den Schwerpunkt auf die Einheitskugel
#         normalized_centroid = centroid / np.linalg.norm(centroid)
#         new_points.append(normalized_centroid)

# print(new_points)

# # x = np.cos(phi) * np.sin(theta)
# # y = np.sin(phi) * np.sin(theta)
# # z = np.cos(theta)


# # fig = plt.figure()
# # ax = fig.add_subplot(projection = "3d")
# # ax.scatter(x, y, z)
# # plt.show()



# phi, theta = fibonacci_sphere_2(1500)

# x = np.cos(phi) * np.sin(theta)
# y = np.sin(phi) * np.sin(theta)
# z = np.cos(theta)


# fig = plt.figure()
# ax = fig.add_subplot(projection = "3d")
# ax.scatter(x, y, z)
# plt.show()

# points = np.array([x, y, z]).T


# from scipy.spatial import Delaunay, Voronoi
# tri = Delaunay(points)

# # Berechne die äquidistanten Winkel in Kugelkoordinaten
# phi = []
# theta = []

# for simplex in tri.simplices:
#     for i in range(3):
#         p = points[simplex[i]]
#         r, azimuth, inclination = np.linalg.norm(p), np.arctan2(p[1], p[0]), np.arccos(p[2] / np.linalg.norm(p))

#         # Füge die Winkel zu den Listen hinzu
#         phi.append(azimuth)
#         theta.append(inclination)