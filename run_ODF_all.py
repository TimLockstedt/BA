import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import vispy_odf



parser = argparse.ArgumentParser(description="number of File, coefficient of ODFs")
parser.add_argument("name", type=str, default=None, help="NAME.npy")
parser.add_argument("coefficient", type=float, default=1)
args = parser.parse_args()
odf_coeff = np.load(f"{args.name}.npy") # Crossing_npsave
# odf_coeff = np.load("ODF_npsave.npy")
print(odf_coeff.shape)
odf_coeff = odf_coeff[:, :, 0, :].squeeze()
print(odf_coeff.shape)

image = vispy_odf.render_scene(odf_coeff*args.coefficient)
plt.imshow(image)
plt.show()
