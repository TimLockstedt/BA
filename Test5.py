import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import vispy_odf


from new_lib_sphere import *
# Kegel länge
range_r = 3
# Anzahl der Bänder der ODFs und AODFs
bands = 10
# Anzahl der Sampling Punkte
number_of_winkel = 1500
# Gaussfunktion Sigma
sigma = 0.5
# Faktor, welche Punkte in die AODFs eingehen
factor_amp = 100
# Faktor für die Größe der AODFs in der Visualisierung
coefficient = 0.25
# ODFs Generieren
field_theta, field_phi = genereate_divergence()
ODFs = odf3.compute(field_theta[:,:,:,None], field_phi[:,:,:,None], np.ones(field_phi[:,:,:,None].shape), bands)[21-range_r-3:20+range_r+3,21-range_r-3:20+range_r+3,21-range_r-3:20+range_r+3,:]
limit_x, limit_y, limit_z = ODFs.shape[0],ODFs.shape[1],ODFs.shape[2] #20,20,20# 
print(ODFs.shape)
# Kegel generieren
phi, theta = fibonacci_sphere(number_of_winkel)
result, basis = get_result_basis(range_r, bands, phi, theta)
result_rot = reverse_rotate_and_translate_data_noTranslation(result, phi, theta)
weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], 0, 0, sigma)


# AODFs Generieren
AODFs = np.array([
    Get_AODF_noRand_noCache_amp(ODFs,result, basis, weights,phi,theta,i,j,k, sigma=sigma, factor_amp=factor_amp, bands=bands)[0]
    for i in tqdm(range(range_r, limit_x - range_r))
    for j in range(range_r, limit_y - range_r)
    for k in range(range_r, limit_z - range_r)
])

# Amps = np.array([
#     Get_AODF_noRand_noCache_amp(ODFs,result, basis, weights,phi,theta,i,j,k, sigma=sigma, factor_amp=factor_amp, bands=bands)[1]
#     for i in tqdm(range(range_r, limit_x - range_r))
#     for j in range(range_r, limit_y - range_r)
#     for k in range(range_r, limit_z - range_r)
# ])
# np.save(f"Amps_Stern_NoRand_nocache_b{bands}_s{sigma*10}_fib_2", Amps)
length = int(np.round(AODFs.shape[0]**(1/3)))
AODFs = np.reshape(AODFs,(length,length,length,odf.get_num_coeff(bands)))
# np.save(f"AODF_Stern_NoRand_nocache_b{bands}_s{sigma*10}_fib_2", AODFs)

# odf.visualize_odf(AODFs[2,2,2,:],120,120)
plt.show()



odf_coeff = AODFs
print(odf_coeff.shape)
odf_coeff = odf_coeff[:, :, 0, :].squeeze()
print(odf_coeff.shape)

image = vispy_odf.render_scene(odf_coeff*coefficient)
plt.imshow(image)

plt.savefig(f"new_lib_halfsphere_with0_fib_b{bands}_s{int(sigma*10)}_c{coefficient}_famp{factor_amp}_n{number_of_winkel}_r{range_r}_fib_2.png", dpi=500)
plt.show()
