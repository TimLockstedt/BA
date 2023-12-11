import h5py
import matplotlib.pyplot as plt
import numpy as np
import vispy_odf


args_name = "9_AODF_npsave_121_new"
args_coefficient = 0.8
odf_coeff = np.load(f"{args_name}.npy") # Crossing_npsave
# odf_coeff = np.load("ODF_npsave.npy")
print(odf_coeff.shape)
odf_coeff = odf_coeff[:, :, 0, :].squeeze()
print(odf_coeff.shape)

image = vispy_odf.render_scene(odf_coeff*args_coefficient)
plt.imshow(image)
plt.show()
