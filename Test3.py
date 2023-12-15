from new_lib import *
range_r = 5
bands = 10
number_of_winkel = 3000
sigma = 0.3


result, basis = get_result_basis(range_r, bands, number_of_winkel)

field_theta, field_phi = genereate_divergence()
ODFs = odf3.compute(field_theta[:,:,:,None], field_phi[:,:,:,None], np.ones(field_phi[:,:,:,None].shape), bands)[21-range_r-3:20+range_r+3,21-range_r-3:20+range_r+3,21-range_r-3:20+range_r+3,:]

limit_x, limit_y, limit_z = ODFs.shape[0],ODFs.shape[1],ODFs.shape[2] #20,20,20# 

result_rot = reverse_rotate_and_translate_data_noTranslation(result, number_of_winkel)
weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], 0, 0, sigma)


phi, theta = fibonacci_sphere(number_of_winkel)
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


# phi, theta = fibonacci_sphere(750)
# phi_, theta_ = fibonacci_sphere_2(750)

# x = np.cos(phi) * np.sin(theta)
# y = np.sin(phi) * np.sin(theta)
# z = np.cos(theta)

# x_ = np.cos(phi_) * np.sin(theta_)
# y_ = np.sin(phi_) * np.sin(theta_)
# z_ = np.cos(theta_)


# fig = plt.figure()
# ax = fig.add_subplot(projection = "3d")
# ax.scatter(x, y, z)
# ax.scatter(x_, y_,z_)
# plt.show()