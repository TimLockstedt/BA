import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from colour import Color
from collections import Counter
import random
from time import gmtime, strftime 
from numba import jit, njit
import pickle 
import odf3
import odf
import h5py
from tqdm import tqdm
import h5py
import vispy_odf


def Get_AODF_noRand_noCache(ODFs:np.ndarray, result:np.ndarray, basis:np.ndarray, weights:np.ndarray, phi:np.ndarray, theta:np.ndarray,x_koord:int, y_koord:int, z_koord:int, factor_amp:int = 1000, sigma:float=2,bands:int = 10):
    # Mit weights alle punkte im Kegel ablaufen
    result_ = np.copy(result)
    result_[:,0,:] += x_koord
    result_[:,1,:] += y_koord
    result_[:,2,:] += z_koord
    AODF_Amplitude = get_amplitude(result_, ODFs, basis, weights)

    # Scatterplot generieren
    # x = AODF_Amplitude[:,None] * np.cos(phi) * np.sin(theta)
    # y = AODF_Amplitude[:,None] * np.sin(phi) * np.sin(theta)
    # z = AODF_Amplitude[:,None] * np.cos(theta)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection = "3d")
    # n = 10000
    # ax.scatter(x[:n],y[:n],z[:n])
    # plt.show()
    
    greater_one = np.where((AODF_Amplitude*factor_amp) > 1)[0]
    multiple_dir = np.repeat(phi[greater_one], np.round((AODF_Amplitude[greater_one]) * factor_amp).astype(int))
    multiple_inc = np.pi/2 - np.repeat(theta[greater_one], np.round((AODF_Amplitude[greater_one]) * factor_amp).astype(int))
    # odf3._get_bands_from_coeff(ODFs.shape[-1])
    nan_mask = ~np.isnan(multiple_inc)
    Test_mask = multiple_inc == 0
    if str(Test_mask[nan_mask].shape) == "(0,)":
        return np.empty((odf.get_num_coeff(bands),))
    AODF_d = odf.compute(np.ravel(multiple_dir[nan_mask]),np.ravel(multiple_inc[nan_mask]), np.ones(np.ravel(multiple_inc[nan_mask]).shape), bands)
    return AODF_d



def get_amplitude(result: np.ndarray, ODFs: np.ndarray, basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    nan_mask = np.isnan(result)
    weights[nan_mask[:,0,:]] = 0
    result_ = np.array(np.copy(result), dtype=int)
    result_[nan_mask] = 0
    Masked_ODFs = ODFs[result_[:,0,:],result_[:,1,:],result_[:,2,:]]
    Amplituden = np.einsum("ni,nj,nji -> n", basis, 
                    weights,
                    Masked_ODFs
                    )/np.sum(~nan_mask[:,0,:], axis=1)
    return Amplituden



def fibonacci_sphere_2(num_pts:int=1000):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    theta = np.arccos(1 - 2*indices/num_pts)
    phi = np.pi * (1 + 5**0.5) * indices
    return (phi, theta)



def alpha_to_phi(alpha:np.ndarray, beta:np.ndarray):
    x = (np.cos(beta))
    y = (np.sin(alpha)*np.sin(beta))
    z = (-np.cos(alpha)*np.sin(beta))
    return np.arctan2(y,x), np.arccos(z)


def phi_to_alpha(phi:np.ndarray, theta:np.ndarray):
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)

    beta = np.arccos(x)
    alpha = np.arctan2(z,y)+math.pi/2
    return alpha, beta


def get_result_basis(range_r, bands, alpha:np.ndarray, beta:np.ndarray):
    print(alpha.shape)
    result = koords_in_kegel(range_r, alpha, beta)
    phi, theta = alpha_to_phi(alpha, beta)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    basis = get_basis(phi, costheta, sintheta, bands)

    return (result, basis)


def koords_in_kegel(range_r = 4, alpha = np.array([]), beta = np.array([])):
    # Winkel umdrehen, da Koordinatensystem gedreht wird anstatt der Punkte
    cos_alpha = np.cos(-alpha)
    sin_alpha = np.sin(-alpha)
    cos_beta = np.cos(-beta)
    sin_beta = np.sin(-beta)
    cos_alpha_cos_beta = cos_alpha*cos_beta
    sin_alpha_cos_beta = sin_alpha*cos_beta
    cos_alpha_sin_beta = cos_alpha*sin_beta
    sin_alpha_sin_beta = sin_alpha*sin_beta
    
    # erstelle das Gitter für x, y und z
    x = np.linspace(-range_r*2, range_r*2, range_r*4+1)
    y = np.copy(x)
    z = np.copy(x)
    length = int(((2*range_r)**3)/6)+1# int((2*range_r)**3)
    print(cos_alpha.shape)
    # Erstelle das Gitter für x, y und z
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
    # test_x = np.copy(x_grid)
    result = np.empty(((cos_alpha.shape[0]), 3, length)) * np.nan

    for i, ca, sa, cb, sb, cacb, sacb, casb, sasb in tqdm(zip(range((cos_alpha.shape[0])), cos_alpha, sin_alpha, cos_beta, sin_beta, cos_alpha_cos_beta, sin_alpha_cos_beta, cos_alpha_sin_beta, sin_alpha_sin_beta)):
        # erst x achse dann y achse
        # Sphere_mask: 
        # mask = ((ca*y_grid-sa*z_grid)**2 + (cacb*z_grid-sb*x_grid+sacb*y_grid)**2 + (casb*z_grid+sasb*y_grid+cb*x_grid)**2) < range_r**2 
        # mask = (
        #         ((ca*y_grid-sa*z_grid)**2 + (cacb*z_grid-sb*x_grid+sacb*y_grid)**2 < (casb*z_grid+sasb*y_grid+cb*x_grid)**2) & 
        #         (0 < (casb*z_grid+sasb*y_grid+cb*x_grid)) & 
        #         (range_r**2 > ((casb*z_grid+sasb*y_grid+cb*x_grid)**2 +(ca*y_grid-sa*z_grid)**2 + (cacb*z_grid-sb*x_grid+sacb*y_grid)**2))
        #         )
        mask = (
                ((ca*y_grid-sa*z_grid)**2 + (cacb*z_grid-sb*x_grid+sacb*y_grid)**2 < (casb*z_grid+sasb*y_grid+cb*x_grid)**2) & 
                (0 < (casb*z_grid+sasb*y_grid+cb*x_grid)) & 
                (range_r > (casb*z_grid+sasb*y_grid+cb*x_grid))
                )
        # x,y,z werte mit der spezifischen Maske auf die Kegel zuschneiden
        x_mask = x_grid[mask]
        y_mask = y_grid[mask]
        z_mask = z_grid[mask]
        x_mask = np.append(x_mask,[0])
        y_mask = np.append(y_mask,[0])
        z_mask = np.append(z_mask,[0])
        data = np.array([x_mask, y_mask, z_mask])
        result[i,:,:np.shape(data)[1]] = data
    return result



def reverse_rotate_and_translate_data_noTranslation(data, alpha:np.ndarray, beta:np.ndarray):
    # Translation Rückgängig machen
    data_ = np.copy(data)
    # Für Optimierung Variablen vor definieren
    cos_alpha = np.cos(-alpha)
    sin_alpha = np.sin(-alpha)
    cos_beta = np.cos(-beta)
    sin_beta = np.sin(-beta)
    # Erzeuge Rotationsmatrizen für alle Winkel gleichzeitig
    rotation_matrix = np.empty((len(alpha), 3, 3))
    # Fülle die Rotationsmatrizen für alle Winkel gleichzeitig
    rotation_matrix[:, 0, 0] = cos_beta.ravel()
    rotation_matrix[:, 0, 1] = (sin_alpha*sin_beta).ravel()
    rotation_matrix[:, 0, 2] = (cos_alpha*sin_beta).ravel()

    rotation_matrix[:, 1, 0] = np.zeros_like(cos_beta).ravel()
    rotation_matrix[:, 1, 1] = cos_alpha.ravel()
    rotation_matrix[:, 1, 2] = -sin_alpha.ravel()

    rotation_matrix[:, 2, 0] = -sin_beta.ravel()
    rotation_matrix[:, 2, 1] = (sin_alpha*cos_beta).ravel()
    rotation_matrix[:, 2, 2] = (cos_alpha*cos_beta).ravel()
    # Wende die Rotationsmatrix auf die Daten an
    result = np.matmul(rotation_matrix, data_)
    return result


def gauss_3d(x,y,z,mu_x=0,mu_y=0,mu_z=0,sigma=2):
    return 1/(np.sqrt(math.pi*2)*sigma)*np.exp(-1/2*(((x-mu_x)/sigma)**2+((y-mu_y)/sigma)**2+((z-mu_z)/sigma)**2))


def gauss_2d(x,y,mu_x=0,mu_y=0,sigma=2):
    # return np.ones(x.shape)
    return 1/(np.sqrt(math.pi*2)*sigma)*np.exp(-1/2*(((x-mu_x)/sigma)**2+((y-mu_y)/sigma)**2))


