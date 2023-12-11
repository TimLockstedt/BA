from main_lib import *

# Cache erstellen
factor = 50
range_r = 3
bands = 10
dict_4_10, dict_basis = generate_dict_basis_new(range_r, factor, bands)
x_koord, y_koord, z_koord = 25,25,25

bands = 10
# Anzahl der Winkel festlegen
number_of_winkel = 1000
# Daten einlesen/Generieren
## Daten Generieren Y-Shape
ODFs = get_Y_Odfs_noise()
sigma = 0.1
factor_amp = 1000
AODFs = np.empty((20,20,20,odf.get_num_coeff(bands)))
for i in range(15,35):
    for j in range(15,35):
        for k in range(15,35):
            AODF_single = Get_AODF(ODFs,dict_4_10, dict_basis,factor,i,j,k, sigma=sigma, factor_amp=factor_amp, number_of_winkel=number_of_winkel)
            AODFs[i,j,k,:] = AODF_single[0,0,0,:]

np.save(f"9_AODF_npsave_121_new", AODFs)
# 6_ODF_fastpli_factor{factor}_ranger{range_r}_sigma{sigma}_factoramp{factor_amp}_numerWinkel{number_of_winkel}_bands{odf.get_num_coeff(bands)}