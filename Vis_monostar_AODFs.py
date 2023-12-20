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


def genereate_divergence_(middle_x:int=4,middle_y:int=4,middle_z:int=4):
    field_phi = np.empty((9,9,2*range_r+1))
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
number_of_winkel = 2250
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
mask = np.ones(field_phi[:,:,:,None].shape)
mask[4,4,:] = 0
ODFs = odf3.compute(field_theta[:,:,:,None], field_phi[:,:,:,None], mask, bands)

limit_x, limit_y, limit_z = ODFs.shape[0],ODFs.shape[1],ODFs.shape[2] #20,20,20# 
print(ODFs.shape)
# Kegel generieren
alpha, beta = fibonacci_sphere_2(number_of_winkel) #get_points_noRand(12, buffer=0.4) 
phi, theta = alpha_to_phi(alpha, beta)
#theta = np.pi/2 - theta
result, basis = get_result_basis(range_r, bands, alpha, beta)
result_rot = reverse_rotate_and_translate_data_noTranslation(result, alpha, beta)
weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], 0, 0, sigma)


# AODFs Generieren
AODFs = np.array([
    Get_AODF_noRand_noCache_amp(ODFs,result, basis, weights,phi,theta,i,j,k, sigma=sigma, factor_amp=factor_amp, bands=bands)[0]
    for i in tqdm(range(range_r, limit_x - range_r),desc='Schleife x', leave=False, ascii="░▒█")
    for j in tqdm(range(range_r, limit_y - range_r),desc='Schleife y', leave=False, ascii=" ▖▘▝▗▚▞█")
    for k in range(range_r, limit_z - range_r)
])

try:
    AODFs = np.reshape(AODFs,(limit_x - 2*range_r,limit_y - 2*range_r,limit_z - 2*range_r, odf.get_num_coeff(bands)))
except:
    print("AODFs reshape failed, error", AODFs.shape)
    shape_x = input("Bitte shape eingeben, erst x, dann y und z \n")
    shape_y = input("y:\n")
    shape_z = input("z:\n")
    AODFs = np.reshape(AODFs, (shape_x, shape_y, shape_z))

for i in tqdm(range(AODFs.shape[2]),desc='Generiere Bilder', leave=False):
    odf_coeff = AODFs
    print(odf_coeff.shape)
    odf_coeff = odf_coeff[:, :, i, :].squeeze()
    print(odf_coeff.shape)

    image = vispy_odf.render_scene(odf_coeff*coefficient)
    plt.imshow(image)
    plt.ylim(325,440)
    plt.xlim(325,445)
    plt.tick_params(axis='x', labelsize=0)
    plt.tick_params(axis='y', labelsize=0)
    plt.savefig(f"new_lib_MonoStar_{i+range_r}_AODF_b{bands}_s{int(sigma*10)}_c{coefficient}_famp{factor_amp}_n{number_of_winkel}_r{range_r}_fib_2.png", dpi=500)
    plt.clf()
