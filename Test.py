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
ODFs = get_Y_Odfs_noise(bands)

AODFs = np.empty((20,20,20,odf.get_num_coeff(bands)))
for i in range(15,35):
    for j in range(15,35):
        for k in range(15,35):
            AODF_single = Get_AODF(ODFs,dict_4_10, dict_basis,factor,i,j,k, sigma=0.1, factor_amp=1000, number_of_winkel=number_of_winkel)
            AODFs[i-15,j-15,k-15,:] = AODF_single[0,0,0,:]

np.save(f"9_AODF_npsave_{odf.get_num_coeff(bands)}", AODFs)
np.save(f"9_ODF_npsave_{odf.get_num_coeff(bands)}", ODFs[15:35,15:35,15:35,:])