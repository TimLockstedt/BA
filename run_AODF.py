import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import vispy_odf

parser = argparse.ArgumentParser(description="number of File, coefficient of ODFs")
parser.add_argument("number", type=int, default=None)
parser.add_argument("coefficient", type=float, default=1)
args = parser.parse_args()
if args.number == None:
    args.number = str(input("Name der Datei: X_npsave_121.npy\n"))

odf_coeff = np.load(f"{args.number}_AODF_npsave_121.npy") # Crossing_npsave
# odf_coeff = np.load("ODF_npsave.npy")
print(odf_coeff.shape)
odf_coeff = odf_coeff[:, :, 0, :].squeeze()
print(odf_coeff.shape)

image = vispy_odf.render_scene(odf_coeff*args.coefficient)
plt.imshow(image)
plt.show()
