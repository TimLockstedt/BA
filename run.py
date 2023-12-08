import h5py
import matplotlib.pyplot as plt
import numpy as np

import vispy_odf

# file = "ODFs_Ycrossing_10band.h5"
# file = "AODFs_Ycrossing_10band.h5"
# with h5py.File(file) as h5f:
#     print(h5f["Image"].shape)
#     odf_coeff = h5f["Image"][:, :, 0, :].squeeze()


odf_coeff = np.load("AODF_npsave_121.npy") # Crossing_npsave
# odf_coeff = np.load("ODF_npsave.npy")
print(odf_coeff.shape)
odf_coeff = odf_coeff[:, :, 0, :].squeeze()
print(odf_coeff.shape)

image = vispy_odf.render_scene(odf_coeff)
plt.imshow(image)
plt.show()
