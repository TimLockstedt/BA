from new_lib import *

# def genereate_divergence_():
#     field_phi = np.empty((50,50,2*range_r+1))
#     field_theta = np.copy(field_phi)
#     for i in range(field_phi.shape[0]):
#         for j in range(field_phi.shape[1]):
#             for k in range(field_phi.shape[2]):
#                 field_phi[i,j,k] = np.arccos((k-25)/np.linalg.norm([i-25,j-25,k-25])) # np.arccos(k+20/np.linalg.norm([i+20,j+20,k+20]))
#                 field_theta[i,j,k] = np.arctan2(j-25,i-25)
#     return(field_theta, field_phi)


def genereate_divergence_(middle_x:int=7,middle_y:int=7,middle_z:int=2):
    field_phi = np.empty((15,15,2*range_r+1))
    field_theta = np.copy(field_phi)
    for i in range(field_phi.shape[0]):
        for j in range(field_phi.shape[1]):
            for k in range(field_phi.shape[2]):
                field_phi[i,j,k] = np.arccos((k-middle_z)/np.linalg.norm([i-middle_x,j-middle_y,k-middle_z])) # np.arccos(k+20/np.linalg.norm([i+20,j+20,k+20]))
                field_theta[i,j,k] = np.arctan2(j-middle_y,i-middle_x)
    return(field_theta, field_phi)


# Kegel länge
range_r = 3
# Anzahl der Bänder der ODFs und AODFs
bands = 10
# Anzahl der Sampling Punkte
number_of_winkel = 1500
# Gaussfunktion Sigma
sigma = 0.3
# Faktor, welche Punkte in die AODFs eingehen
factor_amp = 100
# Faktor für die Größe der AODFs in der Visualisierung
coefficient = 0.7
# ODFs Generieren
# Direction_image, Inclination_image, mask_image, rel_image = load_data(300, 310)
# ODFs = odf3.compute(np.deg2rad(Direction_image)[:,:,:,None], np.deg2rad(Inclination_image)[:,:,:,None], mask_image[:,:,:,None], bands)[600:650,900:950,:,:]

field_theta, field_phi = genereate_divergence_()
# field_theta[:,:,:] += np.pi/2

ODFs = odf3.compute(field_theta[:,:,:,None], field_phi[:,:,:,None], np.ones(field_phi[:,:,:,None].shape), bands)

limit_x, limit_y, limit_z = ODFs.shape[0],ODFs.shape[1],ODFs.shape[2] #20,20,20# 
print(ODFs.shape)
# Kegel generieren
alpha, beta = fibonacci_sphere_2(number_of_winkel) #get_points_noRand(12, buffer=0.4) 
phi, theta = alpha_to_phi(alpha, beta)
#theta = np.pi/2 - theta
result, basis = get_result_basis(range_r, bands, alpha, beta)

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.scatter(*result[0,:,:])
plt.show()