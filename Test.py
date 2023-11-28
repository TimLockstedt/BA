from Test_lib import *
import odf2
import h5py
range_r, factor = 3, 100
dict_10_50 = generate_dict(range_r, factor) 
for i in range(10):
    if i == 0:
        f_mask = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s030{i}_a00_d000_Mask.h5', 'r')
        f_Direction = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s030{i}_ROFL_Direction.h5', 'r')
        f_Inclination = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s030{i}_ROFL_Inclination.h5', 'r')
        f_rel = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s030{i}_ROFL_T_rel.h5', 'r')
        f_mask_image = np.array(f_mask["Image"])[:,:,None]
        f_Direction_image = np.array(f_Direction["Image"])[:,:,None]
        f_Inclination_image = np.array(f_Inclination["Image"])[:,:,None]
        f_rel_image = np.array(f_rel["Image"])[:,:,None]
    else:
        f_mask1 = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s030{i}_a00_d000_Mask.h5', 'r')
        f_Direction1 = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s030{i}_ROFL_Direction.h5', 'r')
        f_Inclination1 = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s030{i}_ROFL_Inclination.h5', 'r')
        f_rel1 = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s030{i}_ROFL_T_rel.h5', 'r')
        f_mask_image1 = np.array(f_mask1["Image"])[:,:,None]
        f_Direction_image1 = np.array(f_Direction1["Image"])[:,:,None]
        f_Inclination_image1 = np.array(f_Inclination1["Image"])[:,:,None]
        f_rel_image1 = np.array(f_rel1["Image"])[:,:,None]
        f_mask_image = np.concatenate((f_mask_image, f_mask_image1), axis=2)
        f_Direction_image = np.concatenate((f_Direction_image, f_Direction_image1), axis=2)
        f_Inclination_image = np.concatenate((f_Inclination_image, f_Inclination_image1), axis=2)
        f_rel_image = np.concatenate((f_rel_image, f_rel_image1), axis=2)

band = 5
ODFs = odf2.compute(f_Direction_image[:,:,:,None], f_Inclination_image[:,:,:,None], f_mask_image[:,:,:,None], band)
x_koord, y_koord, z_koord = 600,600,305
number_of_winkel = 1000

rng = np.random.default_rng(random.randint(100000,10000000000))
beta = np.arccos(1-2*(rng.random(number_of_winkel).reshape(number_of_winkel,1)))
alpha = rng.random(number_of_winkel).reshape(number_of_winkel,1)*math.pi*2

result, phi, costheta, sintheta = kegel_from_dict(dict_10_50, factor, x_koord, y_koord, z_koord, alpha, beta, get_phi_theta=True)
result_rot = reverse_rotate_and_translate_data(result, x_koord, y_koord, z_koord, alpha, beta)

weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], y_koord, z_koord)

def get_basis(phi:np.ndarray, costheta:np.ndarray, sintheta:np.ndarray, band:int)->np.ndarray:
    basis = np.empty((phi.shape[0], odf2.get_num_coeff(band)))
    for i, (p, ct, st) in enumerate(zip(phi, costheta, sintheta)):
        basis[i, :] = odf2._analytic_single_odf(ct, st, p, band)
    return basis
basis = get_basis(phi, costheta, sintheta, band)



AODF_Amplitude = np.empty((result.shape[0])) # np.ones((*ODFs.shape[:3], 66))
for i, res in enumerate(result):
    nan_mask = ~np.isnan(res)
    mask = np.array(res[nan_mask], int)
    mask_3d = np.reshape(mask, (3, int(mask.shape[0]/3)))
    weight = weights[i, nan_mask[0]]
    Test_ODF_masked = np.multiply(weight, ODFs[mask_3d[0],mask_3d[1],mask_3d[2]-300].T).T
    current = np.multiply(basis[i], Test_ODF_masked)
    sum = np.sum(current)
    # if sum != 0.0:
    # print(sum,result.shape[0],res[0,~np.isnan(res[0])].shape[0])
    AODF_Amplitude[i] = sum/res[0,~np.isnan(res[0])].shape[0]
    
x = AODF_Amplitude * np.cos(phi) * sintheta
y = AODF_Amplitude * np.sin(phi) * sintheta
z = AODF_Amplitude * costheta
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.scatter(x,y,z)
plt.show()