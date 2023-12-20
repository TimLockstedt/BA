from new_lib import *

def load_data(start, end):
    for i in range(end-start):
        if i == 0:
            f_mask = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_a00_d000_Mask.h5', 'r')
            f_Direction = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_ROFL_Direction.h5', 'r')
            f_Inclination = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_ROFL_Inclination.h5', 'r')
            f_rel = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_ROFL_T_rel.h5', 'r')
            f_mask_image = np.array(f_mask["Image"])[:,:,None]
            f_Direction_image = np.array(f_Direction["Image"])[:,:,None]
            f_Inclination_image = np.array(f_Inclination["Image"])[:,:,None]
            f_rel_image = np.array(f_rel["Image"])[:,:,None]
        else:
            f_mask1 = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_a00_d000_Mask.h5', 'r')
            f_Direction1 = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_ROFL_Direction.h5', 'r')
            f_Inclination1 = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_ROFL_Inclination.h5', 'r')
            f_rel1 = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_ROFL_T_rel.h5', 'r')
            f_mask_image1 = np.array(f_mask1["Image"])[:,:,None]
            f_Direction_image1 = np.array(f_Direction1["Image"])[:,:,None]
            f_Inclination_image1 = np.array(f_Inclination1["Image"])[:,:,None]
            f_rel_image1 = np.array(f_rel1["Image"])[:,:,None]
            f_mask_image = np.concatenate((f_mask_image, f_mask_image1), axis=2)
            f_Direction_image = np.concatenate((f_Direction_image, f_Direction_image1), axis=2)
            f_Inclination_image = np.concatenate((f_Inclination_image, f_Inclination_image1), axis=2)
            f_rel_image = np.concatenate((f_rel_image, f_rel_image1), axis=2)
    return (f_Direction_image, f_Inclination_image, f_mask_image, f_rel_image)


# Kegel länge
range_r = 3
# Anzahl der Bänder der ODFs und AODFs
bands = 10
# Anzahl der Sampling Punkte
number_of_winkel = 1500
# Gaussfunktion Sigma
sigma = 0.5
# Faktor, welche Punkte in die AODFs eingehen
factor_amp = 10
# Faktor für die Größe der AODFs in der Visualisierung
coefficient = 0.5
# ODFs Generieren
Direction_image, Inclination_image, mask_image, rel_image = load_data(300, 310)
ODFs = odf3.compute(np.deg2rad(Direction_image)[:,:,:,None], np.deg2rad(Inclination_image)[:,:,:,None], mask_image[:,:,:,None], bands)[600:650,900:950,:,:]
limit_x, limit_y, limit_z = ODFs.shape[0],ODFs.shape[1],ODFs.shape[2] #20,20,20# 
print(ODFs.shape)
# Kegel generieren
alpha, beta = fibonacci_sphere(number_of_winkel) #get_points_noRand(12, buffer=0.4) 
phi, theta = alpha_to_phi(alpha, beta)
#theta = np.pi/2 - theta
result, basis = get_result_basis(range_r, bands, alpha, beta)
result_rot = reverse_rotate_and_translate_data_noTranslation(result, alpha, beta)
weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], 0, 0, sigma)


# AODFs Generieren
AODFs = np.array([
    Get_AODF_noRand_noCache_amp(ODFs,result, basis, weights,phi,theta,i,j,k, sigma=sigma, factor_amp=factor_amp, bands=bands)[0]
    for i in tqdm(range(range_r, limit_x - range_r),desc='Schleife 1', leave=False)
    for j in tqdm(range(range_r, limit_y - range_r),desc='Schleife 2', leave=False)
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

for i in range(AODFs.shape[2]):
    odf_coeff = AODFs
    print(odf_coeff.shape)
    odf_coeff = odf_coeff[:, :, i, :].squeeze()
    print(odf_coeff.shape)

    image = vispy_odf.render_scene(odf_coeff*coefficient)
    plt.imshow(image)
    plt.savefig(f"new_lib_PLI_600-650_900-950_{300+i+range_r}_AODF_b{bands}_s{int(sigma*10)}_c{coefficient}_famp{factor_amp}_n{number_of_winkel}_r{range_r}_fib_2.png", dpi=500)
    plt.clf()
