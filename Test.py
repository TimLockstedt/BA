from sphere_lib import *


# Cache erstellen
factor = 100
range_r = 3
bands = 10
dict_res, dict_basis = load_dict(range_r, factor), load_dict(range_r, bands)

field_theta, field_phi = genereate_divergence()


# result, basis, phi, theta = kegel_from_dict_withBasis_noTranslation(dict_res, dict_basis, factor, alpha[:,None], beta[:,None], True)
sigma = 0.1
factor_amp = 10
bands = 10
number = 1500


ODFs = odf3.compute(field_theta[:,:,:,None], field_phi[:,:,:,None], np.ones(field_phi[:,:,:,None].shape), 10)[15:26,15:26,15:26,:]

limit_x, limit_y, limit_z = ODFs.shape[0],ODFs.shape[1],ODFs.shape[2] #20,20,20# 
sigma = 0.1
factor_amp = 10
bands = 10
number = 1500
alpha, beta = fibonacci_sphere(number)

result, basis, phi, theta = kegel_from_dict_withBasis_noTranslation(dict_res, dict_basis, factor, alpha[:,None], beta[:,None], True)
result_rot = reverse_rotate_and_translate_data_noTranslation(result, alpha, beta)
weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], 0, 0, sigma)

# AODF = Get_AODF_noRand_noCache(ODFs,result, basis, weights,phi,theta,1,1,0, sigma=sigma, factor_amp=factor_amp)

AODFs = np.array([
    Get_AODF_noRand_noCache(ODFs,result, basis, weights,phi,theta,i,j,k, sigma=sigma, factor_amp=factor_amp)
    for i in tqdm(range(range_r + 1, limit_x - range_r - 1))
    for j in range(range_r + 1, limit_y - range_r - 1)
    for k in range(range_r + 1, limit_z - range_r - 1)
])
np.save(f"AODF_Kugel_Stern_NoRand", np.reshape(AODFs,(limit_x - 2*(range_r + 1),limit_y - 2*(range_r + 1),limit_z - 2*(range_r + 1),121)))



AODFs = np.reshape(AODFs,(limit_x - 2*(range_r + 1),limit_y - 2*(range_r + 1),limit_z - 2*(range_r + 1),121))
odf.visualize_odf(AODFs[1,1,1,:],120,120)
plt.show()
