from main_lib import *

def genereate_swirls():
    field_phi = np.empty((60,60,60))
    field_theta = np.copy(field_phi)
    for i in range(field_phi.shape[0]):
        for j in range(field_phi.shape[1]):
            for k in range(field_phi.shape[2]):
                field_phi[i,j,k] = 0 # np.arccos(k+20/np.linalg.norm([i+20,j+20,k+20])) # np.arccos(k+20/np.linalg.norm([i+20,j+20,k+20]))
                field_theta[i,j,k] = np.pi * (i+j)*0.02
    return(field_theta, field_phi)

field_theta, field_phi = genereate_swirls()
ODFs = odf3.compute(field_theta[:,:,:,None], field_phi[:,:,:,None], np.ones(field_phi[:,:,:,None].shape), 10)

factor = 50
range_r = 3
bands = 10
dict_res, dict_basis = generate_dict_basis_new(range_r, factor, bands)#load_dict(range_r, factor), load_dict(range_r, factor+1)

# Anzahl der Winkel festlegen
number_of_winkel = 1000

limit_x, limit_y, limit_z = 20,20,20 # ODFs.shape[0],ODFs.shape[1],ODFs.shape[2]
sigma = 0.1
factor_amp = 1000
AODFs = np.empty((limit_x-2*range_r,limit_y-2*range_r,limit_z-2*range_r,odf.get_num_coeff(bands)))
for i in range(range_r+1,limit_x-range_r-1):
    for j in range(range_r+1,limit_y-range_r-1):
        for k in range(range_r+1,limit_z-range_r-1):
            AODF_single = Get_AODF(ODFs,dict_res, dict_basis,factor,i,j,k, sigma=sigma, factor_amp=factor_amp, number_of_winkel=number_of_winkel)
            AODFs[i-(range_r+1),j-(range_r+1),k-(range_r+1),:] = AODF_single[0,0,0,:]

np.save(f"Konsistenz_check_small_Rand_6_f{factor}_n{number_of_winkel}", AODFs)
