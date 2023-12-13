from concurrent.futures import ThreadPoolExecutor
import copy


import numpy as np
import math
import matplotlib.pyplot as plt
from colour import Color
from collections import Counter
import random
from time import gmtime, strftime
from numba import jit, njit
import pickle 
import odf3
import odf
import h5py

def reverse_rotate_and_translate_data(data, x_koord = 0, y_koord = 0, z_koord = 0, alpha = np.array([]), beta = np.array([])):
    # Translation Rückgängig machen
    data_ = np.copy(data)
    data_[:,0,:] -= x_koord
    data_[:,1,:] -= y_koord
    data_[:,2,:] -= z_koord
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
    result[:,0,:] += x_koord
    result[:,1,:] += y_koord
    result[:,2,:] += z_koord
    return result

def kegel_from_dict_withBasis(dict_cache:dict, dict_cache_basis:dict, factor:int=10, x_koord:int=0, y_koord:int=0, z_koord:int=0, alpha:np.ndarray=np.array([]), beta:np.ndarray=np.array([]), get_phi_theta = False):
    # Winkelfunktionen vordefinieren
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    # Karthesische Koordinate der symetrieachse berehnen
    x = (cos_beta)
    y = (sin_alpha*sin_beta)
    z = (-cos_alpha*sin_beta)
    koords = np.concatenate((x,y,z), axis=1)
    koords_rounded_int = np.array(np.round(koords*factor, 0), dtype=int)
    # Daten aus dem Dict herausholen
    arrays = [dict_cache[tuple(key)] for key in koords_rounded_int]
    result = np.array(arrays)
    arrays_basis = [dict_cache_basis[tuple(key)] for key in koords_rounded_int]
    res_basis = np.array(arrays_basis)
    # else:  # langsamer
    #     result = np.array(Parallel(n_jobs=-1)(delayed(get_array)(dict_cache,key) for key in koords_rounded_int))

    result[:,0,:] += x_koord
    result[:,1,:] += y_koord
    result[:,2,:] += z_koord
    if get_phi_theta == True:
        # Phi und Theta berechnen für ODFs
        phi = np.arccos(z)
        theta = np.arctan2(y,x)
        return (result, res_basis, phi, theta)

    return result, res_basis

def koords_in_kegel_cache(range_r = 4, alpha = np.array([]), beta = np.array([])):
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
    length = int(((2*range_r)**3)/6)+1

    # Erstelle das Gitter für x, y und z
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
    # test_x = np.copy(x_grid)
    result = np.empty((len(cos_alpha), 3, length)) * np.nan

    for i, ca, sa, cb, sb, cacb, sacb, casb, sasb in zip(range(len(cos_alpha)), cos_alpha, sin_alpha, cos_beta, sin_beta, cos_alpha_cos_beta, sin_alpha_cos_beta, cos_alpha_sin_beta, sin_alpha_sin_beta):
        # erst x achse dann y achse
        mask = (((ca*y_grid-sa*z_grid)**2 + (cacb*z_grid-sb*x_grid+sacb*y_grid)**2 < (casb*z_grid+sasb*y_grid+cb*x_grid)**2) & (0 < (casb*z_grid+sasb*y_grid+cb*x_grid)) & (range_r > (casb*z_grid+sasb*y_grid+cb*x_grid)))
        # x,y,z werte mit der spezifischen Maske auf die Kegel zuschneiden
        x_mask = x_grid[mask]
        y_mask = y_grid[mask]
        z_mask = z_grid[mask]
        data = np.array([x_mask, y_mask, z_mask])
        result[i,:,:np.shape(data)[1]] = data
    return result

def get_basis_xyz_new(x:int, y:int, z:int, band:int=10, factor:int=10):
    phi = np.arccos(z/factor)
    theta = np.arctan2(y,x)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    basis = np.empty((phi.shape[0], odf3.get_num_coeff(band)))
    for i, (p, ct, st) in enumerate(zip(phi, costheta, sintheta)):
        basis[i, :] = odf3._analytic_single_odf(ct, st, p, band)
    return basis

def get_points_noRand(range_r, buffer = 1):
    # Punkte vordefinieren
    x = np.arange(-range_r, range_r+1, 1)
    y = np.copy(x)
    z = np.copy(x)
    # erstelle das Gitter für x, y und z
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
    # Punkte außerhalb des Kegels filtern 
    mask = (x_grid**2 + y_grid**2 + z_grid**2 <= (range_r+buffer)**2) & (x_grid**2 + y_grid**2 + z_grid**2 >= (range_r-buffer)**2)
    x_mask = x_grid[mask]
    y_mask = y_grid[mask]
    z_mask = z_grid[mask]
    # x, y, z zusammenfassen
    Kugelschale = np.concatenate((x_mask[:,None], y_mask[:,None], z_mask[:,None]), axis=1)
    # Kugelkoordinaten Winkel bestimmen
    beta = np.arccos(Kugelschale[:,0]/range_r)
    if np.sum(np.isnan(beta)) > 0:
        print(f"Warning, {np.sum(np.isnan(beta))} nan entries in beta!!!")
    alpha = np.arctan2(Kugelschale[:,2],Kugelschale[:,1])+math.pi/2
    if np.sum(np.isnan(alpha)) > 0:
        print(f"Warning, {np.sum(np.isnan(alpha))} nan entries in alpha!!!")
    return(Kugelschale, alpha, beta)

def generate_dict_basis_new_noRand(range_r:int = 10, factor:int = 10, band: int = 10, buffer: int = 0.8):
    Kugelschale, alpha, beta = get_points_noRand(factor, buffer)
    temp_dict = {tuple(a_row): b_row for a_row, b_row in zip(Kugelschale, koords_in_kegel_cache(range_r, alpha, beta))}
    temp_dict_basis = {tuple(a_row): b_row for a_row, b_row in zip(Kugelschale, get_basis_xyz_new(*Kugelschale.T, band, factor))}
    return temp_dict, temp_dict_basis, alpha, beta

def Get_AODF_noRand(ODFs:np.ndarray, dict_res:dict, dict_basis:dict, factor:int, x_koord:int, y_koord:int, z_koord:int, alpha:np.ndarray, beta:np.ndarray, factor_amp:int = 1000, sigma:float=2):
    # # Winkel generieren
    # rng = np.random.default_rng(random.randint(100000,10000000000))
    # beta = np.arccos(1-2*(rng.random(number_of_winkel).reshape(number_of_winkel,1)))
    # alpha = rng.random(number_of_winkel).reshape(number_of_winkel,1)*math.pi*2

    # Punkte im Kegel mit Basisvektoren und Winkeln generieren
    result, basis, phi, theta = kegel_from_dict_withBasis(dict_res, dict_basis, factor, x_koord, y_koord, z_koord, alpha, beta, True)
    result_rot = reverse_rotate_and_translate_data(result, x_koord, y_koord, z_koord, alpha, beta)
    weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], y_koord, z_koord, sigma)

    # Mit weights alle punkte im Kegel ablaufen
    AODF_Amplitude = get_amplitude(result, ODFs, basis, weights)
    
    greater_one = np.where((AODF_Amplitude*factor_amp) > 1)[0]
    multiple_dir = np.repeat(phi[greater_one], np.round((AODF_Amplitude[greater_one]) * factor_amp).astype(int))
    multiple_inc = np.pi/2 - np.repeat(theta[greater_one], np.round((AODF_Amplitude[greater_one]) * factor_amp).astype(int))
    bands = 10
    # odf3._get_bands_from_coeff(ODFs.shape[-1])
    nan_mask = ~np.isnan(multiple_inc)
    Test_mask = multiple_inc == 0
    if str(Test_mask[nan_mask].shape) == "(0,)":
        return np.empty((1,1,1,odf.get_num_coeff(bands)))
    AODF_d = odf.compute(np.ravel(multiple_dir[nan_mask])[None,None,None,:],np.ravel(multiple_inc[nan_mask])[None,None,None,:], np.ones(np.ravel(multiple_inc[nan_mask])[None,None,None,:].shape), bands)
    return AODF_d


def get_amplitude(result: np.ndarray, ODFs: np.ndarray, basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.array([
        (
            np.dot(
                basis[None, i],
                np.dot(
                    weights[i, ~np.isnan(res)[0]][None, :],
                    ODFs[
                        tuple(np.reshape(np.array(res[~np.isnan(res)], int), (3, -1)))
                    ]
                ).T
            ).item() / np.reshape(np.array(res[~np.isnan(res)], int), (3, -1)).shape[-1]
        ) if np.reshape(np.array(res[~np.isnan(res)], int), (3, -1)).shape[-1] > 0
        else 0
        for i, res in enumerate(result)
    ])

                    # ODFs[
                    #     np.reshape(np.array(res[~np.isnan(res)], int), (3, -1))
                    # ]   


def genereate_swirls():
    field_phi = np.empty((60,60,60))
    field_theta = np.copy(field_phi)
    for i in range(field_phi.shape[0]):
        for j in range(field_phi.shape[1]):
            for k in range(field_phi.shape[2]):
                field_phi[i,j,k] = 0 # np.arccos(k+20/np.linalg.norm([i+20,j+20,k+20])) # np.arccos(k+20/np.linalg.norm([i+20,j+20,k+20]))
                field_theta[i,j,k] = np.pi * (i+j)*0.02
    return(field_theta, field_phi)


def gauss_2d(x,y,mu_x=0,mu_y=0,sigma=2):
    return 1/(np.sqrt(math.pi*2)*sigma)*np.exp(-1/2*(((x-mu_x)/sigma)**2+((y-mu_y)/sigma)**2))


# Cache erstellen
factor = 7
range_r = 3
bands = 10
dict_res, dict_basis, alpha, beta = generate_dict_basis_new_noRand(range_r, factor, bands, buffer=0.4)#load_dict(range_r, factor), load_dict(range_r, factor+1)
alpha.shape
#field_theta, field_phi = genereate_divergence()
field_theta, field_phi = genereate_swirls()
ODFs = odf3.compute(field_theta[:,:,:,None], field_phi[:,:,:,None], np.ones(field_phi[:,:,:,None].shape), 10)


sigma = 0.1
factor_amp = 1000
limit_x, limit_y, limit_z = 20,20,20# ODFs.shape[0],ODFs.shape[1],ODFs.shape[2]
bands = 10

num_components = odf.get_num_coeff(bands)
def parallel_compute_AODFs(ODFs, dict_res, dict_basis, factor, alpha, beta, sigma, factor_amp, i, j, k):
    local_ODFs = copy.deepcopy(ODFs)
    local_dict_res = copy.deepcopy(dict_res)
    local_dict_basis = copy.deepcopy(dict_basis)
    local_factor = copy.deepcopy(factor)
    local_alpha = copy.deepcopy(alpha)
    local_beta = copy.deepcopy(beta)
    local_sigma = copy.deepcopy(sigma)
    local_factor_amp = copy.deepcopy(factor_amp)

    AODF_single = Get_AODF_noRand(local_ODFs, local_dict_res, local_dict_basis, local_factor, i, j, k, alpha=local_alpha[:, None], beta=local_beta[:, None], sigma=local_sigma, factor_amp=local_factor_amp)
    
    return i - range_r - 1, j - range_r - 1, k - range_r - 1, AODF_single[0, 0, 0, :]

# Set the number of threads according to your needs
num_threads = 12

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    AODFs_list = list(executor.map(
        lambda args: parallel_compute_AODFs(ODFs, dict_res, dict_basis, factor, alpha, beta, sigma, factor_amp, *args),
        [(i, j, k) for i in range(range_r + 1, limit_x - range_r - 1) for j in range(range_r + 1, limit_y - range_r - 1) for k in range(range_r + 1, limit_z - range_r - 1)]
    ))

# Convert the list to a numpy array
AODFs = np.zeros((limit_x - range_r - 1, limit_y - range_r - 1, limit_z - range_r - 1, num_components))
for i, j, k, AODF_single in AODFs_list:
    AODFs[i, j, k, :] = AODF_single


np.save("parallel_Test", ADOFs)