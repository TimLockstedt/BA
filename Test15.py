from main_lib_new import *

# Kegel länge
range_r = 3
# Anzahl der Bänder der ODFs und AODFs
bands = 10
# Anzahl der Sampling Punkte
number_of_winkel = 200000
# Gaussfunktion Sigma
sigma = 0.3
# Faktor, welche Punkte in die AODFs eingehen
factor_amp = 100
# Faktor für die Größe der AODFs in der Visualisierung
coefficient = 0.7
# ODFs Generieren

# Cache erstellen
factor = 100
dict_4_10, dict_basis = load_dict(range_r, factor), load_dict(range_r, bands) #generate_dict_basis(range_r, factor, bands)# 
# save_dict(dict_4_10, range_r, factor)
# save_dict(dict_basis, range_r, bands)

xkoord, ykoord, zkoord = 0,0,0
rng = np.random.default_rng(random.randint(100000,10000000000))
beta = np.arccos(1-2*(rng.random(number_of_winkel).reshape(number_of_winkel,1)))
alpha = rng.random(number_of_winkel).reshape(number_of_winkel,1)*math.pi*2

result, basis = kegel_from_dict_withBasis(dict_4_10, dict_basis,factor,xkoord, ykoord, zkoord,alpha, beta)

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.scatter(*result[0,:,:])
plt.show()

# np.concatenate
# def genereate_divergence_():
#     field_phi = np.empty((15,15,2*range_r+1))
#     field_theta = np.copy(field_phi)
#     for i in range(field_phi.shape[0]):
#         for j in range(field_phi.shape[1]):
#             for k in range(field_phi.shape[2]):
#                 field_phi[i,j,k] = np.arccos((k-7)/np.linalg.norm([i-7,j-7,k-7])) # np.arccos(k+20/np.linalg.norm([i+20,j+20,k+20]))
#                 field_theta[i,j,k] = np.arctan2(j-7,i-7)
#     return(field_theta, field_phi)

# field_theta, field_phi = genereate_divergence_()
# # field_theta[:,:,:] += np.pi/2

# ODFs = odf3.compute(field_theta[:,:,:,None], field_phi[:,:,:,None], np.ones(field_phi[:,:,:,None].shape), bands)
# limit_x, limit_y, limit_z = ODFs.shape[0],ODFs.shape[1],ODFs.shape[2] #20,20,20# 

# AODFs = np.array([
#     Get_AODF(ODFs,dict_4_10, dict_basis,factor,i,j,k, sigma=sigma, factor_amp=factor_amp, number_of_winkel=number_of_winkel)
#     for i in tqdm(range(range_r, limit_x - range_r),desc='Schleife x', leave=False, ascii="░▒█")
#     for j in tqdm(range(range_r, limit_y - range_r),desc='Schleife y', leave=False, ascii=" ▖▘▝▗▚▞█")
#     for k in range(range_r, limit_z - range_r)
# ])


# try:
#     AODFs = np.reshape(AODFs,(limit_x - 2*range_r,limit_y - 2*range_r,limit_z - 2*range_r, odf.get_num_coeff(bands)))
# except:
#     print("AODFs reshape failed, error", AODFs.shape)
#     shape_x = input("Bitte shape eingeben, erst x, dann y und z \n")
#     shape_y = input("y:\n")
#     shape_z = input("z:\n")
#     AODFs = np.reshape(AODFs, (shape_x, shape_y, shape_z))

# for i in tqdm(range(AODFs.shape[2]),desc='Generiere Bilder', leave=False):
#     odf_coeff = AODFs
#     print(odf_coeff.shape)
#     odf_coeff = odf_coeff[:, :, i, :].squeeze()
#     print(odf_coeff.shape)

#     image = vispy_odf.render_scene(odf_coeff*coefficient)
#     plt.imshow(image)
#     plt.savefig(f"new_lib_ministar_{i+range_r}_AODF_b{bands}_s{int(sigma*10)}_c{coefficient}_famp{factor_amp}_n{number_of_winkel}_r{range_r}_Rand_{factor}_2pi-theta.png", dpi=500)
#     plt.clf()

