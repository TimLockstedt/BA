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
        mask = (
                ((ca*y_grid-sa*z_grid)**2 + (cacb*z_grid-sb*x_grid+sacb*y_grid)**2 < (casb*z_grid+sasb*y_grid+cb*x_grid)**2) & 
                (0 < (casb*z_grid+sasb*y_grid+cb*x_grid)) & 
                (range_r**2 > ((casb*z_grid+sasb*y_grid+cb*x_grid)**2 +(ca*y_grid-sa*z_grid)**2 + (cacb*z_grid-sb*x_grid+sacb*y_grid)**2))
                )
        # mask = (
        #         ((ca*y_grid-sa*z_grid)**2 + (cacb*z_grid-sb*x_grid+sacb*y_grid)**2 < (casb*z_grid+sasb*y_grid+cb*x_grid)**2) & 
        #         (0 < (casb*z_grid+sasb*y_grid+cb*x_grid)) & 
        #         (range_r > (casb*z_grid+sasb*y_grid+cb*x_grid))
        #         )
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




def koords_in_kegel_MathKegel(range_r = 4, alpha = np.array([]), beta = np.array([])):
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



# anpassung der schnellen ungenauen berechnung der Kegel, da die Funktion zur bewertung der Punlte kontinuiertlich ist.
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


def get_basis(phi:np.ndarray, costheta:np.ndarray, sintheta:np.ndarray, band:int)->np.ndarray:
    basis = np.empty((phi.shape[0], odf3.get_num_coeff(band)))
    for i, (p, ct, st) in tqdm(enumerate(zip(phi, costheta, sintheta))):
        basis[i, :] = odf3._analytic_single_odf(ct, st, p, band)
    return basis



def gauss_3d(x,y,z,mu_x=0,mu_y=0,mu_z=0,sigma=2):
    return 1/(np.sqrt(math.pi*2)*sigma)*np.exp(-1/2*(((x-mu_x)/sigma)**2+((y-mu_y)/sigma)**2+((z-mu_z)/sigma)**2))


def gauss_2d(x,y,mu_x=0,mu_y=0,sigma=2):
    # return np.ones(x.shape)
    return 1/(np.sqrt(math.pi*2)*sigma)*np.exp(-1/2*(((x-mu_x)/sigma)**2+((y-mu_y)/sigma)**2))


def gauss_1d(x,mu_x=0,sigma=2):
    return 1/(np.sqrt(math.pi*2)*sigma)*np.exp(-1/2*(((x-mu_x)/sigma)**2))


def get_result_basis(range_r, bands, alpha:np.ndarray, beta:np.ndarray):
    print(alpha.shape)
    result = koords_in_kegel(range_r, alpha, beta)
    phi, theta = alpha_to_phi(alpha, beta)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    basis = get_basis(phi, costheta, sintheta, bands)

    return (result, basis)

def get_result_basis_MathKegel(range_r, bands, alpha:np.ndarray, beta:np.ndarray):
    print(alpha.shape)
    result = koords_in_kegel_MathKegel(range_r, alpha, beta)
    phi, theta = alpha_to_phi(alpha, beta)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    basis = get_basis(phi, costheta, sintheta, bands)

    return (result, basis)



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
        phi, theta = alpha_to_phi(alpha, beta)
        return (result, res_basis, phi, theta)

    return result, res_basis


# def get_amplitude(result: np.ndarray, ODFs: np.ndarray, basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
#     return np.array([
#         (np.sum(
#             np.dot(
#                 basis[None, i],
#                 np.dot(
#                     weights[i, ~np.isnan(res)[0]][None, :],
#                     ODFs[
#                         tuple(np.reshape(np.array(res[~np.isnan(res)], int), (3, -1)))
#                     ]
#                 ).T
#             ).item() / np.reshape(np.array(res[~np.isnan(res)], int), (3, -1)).shape[-1]
#             ) 
#         )
#         for i, res in enumerate(result)
#     ])

# def get_amplitude(result: np.ndarray, ODFs: np.ndarray, basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
#     return np.array([
#         (np.einsum('ij,ik,kj->', basis[None, i], 
#                     weights[i, ~np.isnan(res)[0]][None, :],
#                     ODFs[
#                         tuple(np.reshape(np.array(res[~np.isnan(res)], int), (3, -1)))
#                     ]
#                     )/ np.reshape(np.array(res[~np.isnan(res)], int), (3, -1)).shape[-1]
#         )
#         for i, res in enumerate(result)
    # ])


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


def get_basis_xyz_new(x:int, y:int, z:int, band:int=10, factor:int=10):
    phi = np.arccos(z/factor)
    theta = np.arctan2(y,x)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    basis = np.empty((phi.shape[0], odf3.get_num_coeff(band)))
    for i, (p, ct, st) in tqdm(enumerate(zip(phi, costheta, sintheta))):
        basis[i, :] = odf3._analytic_single_odf(ct, st, p, band)
    return basis


def generate_dict_basis_new(range_r:int = 10, factor:int = 10, band: int = 10):
    Kugelschale, alpha, beta = get_points(factor)
    temp_dict = {tuple(a_row): b_row for a_row, b_row in zip(Kugelschale, koords_in_kegel_cache(range_r, alpha, beta))}
    temp_dict_basis = {tuple(a_row): b_row for a_row, b_row in zip(Kugelschale, get_basis_xyz_new(*Kugelschale.T, band, factor))}
    return temp_dict, temp_dict_basis


def get_Y_Odfs_old(bands:int=10):
    Direction_zero = np.zeros((100,100,100))
    Inclination_zero = np.copy(Direction_zero)
    mask = np.ones((100,100,100))
    for i in range(Direction_zero.shape[0]):
        for j in range(Direction_zero.shape[1]):
            for k in range(Direction_zero.shape[2]):
                if j <= 25 and i <= 27 and i >= 23 and k <= 50:
                    Direction_zero[i,j,k] = np.pi*(1/2)
                elif i <= 50 and j <= 50 and k <= 50 and j <= -i+52 and j >= -i+48 and j > 25:
                    Direction_zero[i,j,k] = np.pi*(1+1/4)
                elif i <= 50 and j <= 50 and k <= 50 and j <= i+2 and j >= i-2 and j > 25:
                    Direction_zero[i,j,k] = np.pi*(1-1/4)
                else:
                    mask[i,j,k] = 0
    _ODFs = odf3.compute(Direction_zero[:,:,:,None], Inclination_zero[:,:,:,None], mask[:,:,:,None], bands)
    return _ODFs


def Get_AODF(ODFs:np.ndarray, dict_res:dict, dict_basis:dict, factor:int, x_koord:int, y_koord:int, z_koord:int, number_of_winkel:int = 1000, factor_amp:int = 1000, sigma:float=2):
    # Winkel generieren
    rng = np.random.default_rng(random.randint(100000,10000000000))
    beta = np.arccos(1-2*(rng.random(number_of_winkel).reshape(number_of_winkel,1)))
    alpha = rng.random(number_of_winkel).reshape(number_of_winkel,1)*math.pi*2

    # Punkte im Kegel mit Basisvektoren und Winkeln generieren
    result, basis, phi, theta = kegel_from_dict_withBasis(dict_res, dict_basis, factor, x_koord, y_koord, z_koord, alpha, beta, True)
    result_rot = reverse_rotate_and_translate_data(result, x_koord, y_koord, z_koord, alpha, beta)
    weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], y_koord, z_koord, sigma)

    # Mit weights alle punkte im Kegel ablaufen
    AODF_Amplitude = get_amplitude(result, ODFs, basis, weights)

    # Scatterplot generieren
    x = AODF_Amplitude[:,None] * np.cos(phi) * np.sin(theta)
    y = AODF_Amplitude[:,None] * np.sin(phi) * np.sin(theta)
    z = AODF_Amplitude[:,None] * np.cos(theta)
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    n = 10000
    ax.scatter(x[:n],y[:n],z[:n])
    plt.show()


    # num_greater = np.sum(np.rint(AODF_Amplitude[AODF_Amplitude > 0]*100))
    
    # num_greater = int(np.sum(np.rint(AODF_Amplitude[AODF_Amplitude > 0]*factor_amp)))

    # multiple_dir = np.empty((num_greater,1))*np.nan
    # multiple_inc = np.empty((num_greater,1))*np.nan

    # count = 0
    # for _i,_j in enumerate(AODF_Amplitude):
    #     for k in range(int(_j*factor_amp if _j > 0 else 0)):
    #         multiple_dir[count,:] = phi[int(_i)]
    #         multiple_inc[count,:] = np.pi/2 - theta[int(_i)] 
    #         count += 1
    
    greater_one = np.where((AODF_Amplitude*factor_amp) > 1)[0]
    multiple_dir = np.repeat(phi[greater_one], np.round((AODF_Amplitude[greater_one]) * factor_amp).astype(int))
    multiple_inc = np.pi/2 - np.repeat(theta[greater_one], np.round((AODF_Amplitude[greater_one]) * factor_amp).astype(int))
    bands = odf3._get_bands_from_coeff(ODFs.shape[-1])
    nan_mask = ~np.isnan(multiple_inc)
    Test_mask = multiple_inc == 0
    if str(Test_mask[nan_mask].shape) == "(0,)":
        return np.empty((1,1,1,odf.get_num_coeff(bands)))
    AODF_d = odf.compute(np.ravel(multiple_dir[nan_mask])[None,None,None,:],np.ravel(multiple_inc[nan_mask])[None,None,None,:], np.ones(np.ravel(multiple_inc[nan_mask])[None,None,None,:].shape), bands)
    return AODF_d

    # return np.empty((1,1,1,odf.get_num_coeff(bands)))



def get_Y_Odfs(bands:int=10):
    Direction_zero = np.zeros((100,100,100))
    Inclination_zero = np.copy(Direction_zero)
    mask = np.ones((100,100,100))
    for i in range(Direction_zero.shape[0]):
        for j in range(Direction_zero.shape[1]):
            for k in range(Direction_zero.shape[2]):
                if j <= 25 and i <= 27 and i >= 23 and k <= 50:
                    Direction_zero[i,j,k] = np.pi*(1/2)
                elif i <= 50 and j <= 50 and k <= 50 and j <= -i+52 and j >= -i+48 and j > 25:
                    Direction_zero[i,j,k] = np.pi*(1-1/4)
                elif i <= 50 and j <= 50 and k <= 50 and j <= i+2 and j >= i-2 and j > 25:
                    Direction_zero[i,j,k] = np.pi*(1+1/4)
                else:
                    mask[i,j,k] = 0
    _ODFs = odf3.compute(Direction_zero[:,:,:,None], Inclination_zero[:,:,:,None], mask[:,:,:,None], bands)
    return _ODFs


def get_Y_Odfs_noise(bands:int=10):
    Direction_zero = direction = np.random.uniform(0, 1, (40,40,40))*np.pi
    Inclination_zero = np.zeros((40,40,40))
    mask = np.ones((40,40,40))
    for i in range(Direction_zero.shape[0]):
        for j in range(Direction_zero.shape[1]):
            for k in range(Direction_zero.shape[2]):
                if j <= 25 and i <= 27 and i >= 23 and k <= 50:
                    Direction_zero[i,j,k] = np.pi*(1/2)
                elif i <= 50 and j <= 50 and k <= 50 and j <= -i+52 and j >= -i+48 and j > 25:
                    Direction_zero[i,j,k] = np.pi*(1-1/4)
                elif i <= 50 and j <= 50 and k <= 50 and j <= i+2 and j >= i-2 and j > 25:
                    Direction_zero[i,j,k] = np.pi*(1+1/4)
                # else:
                #     mask[i,j,k] = 0
    _ODFs = odf3.compute(Direction_zero[:,:,:,None], Inclination_zero[:,:,:,None], mask[:,:,:,None], bands)
    return _ODFs



def load_data(start, end):
    for i in range(end-start):
        if i == 0:
            f_mask = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_a00_d000_Mask.h5', 'r')
            f_Direction = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_ROFL_Direction.h5', 'r')
            f_Inclination = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_ROFL_Inclination.h5', 'r')
            f_rel = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_ROFL_T_rel.h5', 'r')
            f_mask_image = np.array(f_mask["Image"])[:,:,None]
            f_Direction_image = np.array(f_Direction["Image"])[:,:,None]
            f_Inclination_image = np.array(f_Inclination["Image"])[:,:,None]
            f_rel_image = np.array(f_rel["Image"])[:,:,None]
        else:
            f_mask1 = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_a00_d000_Mask.h5', 'r')
            f_Direction1 = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_ROFL_Direction.h5', 'r')
            f_Inclination1 = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_ROFL_Inclination.h5', 'r')
            f_rel1 = h5py.File(f'MSA_hdf5\MSA0309_M8_70mu_70ms_s0{start+i}_ROFL_T_rel.h5', 'r')
            f_mask_image1 = np.array(f_mask1["Image"])[:,:,None]
            f_Direction_image1 = np.array(f_Direction1["Image"])[:,:,None]
            f_Inclination_image1 = np.array(f_Inclination1["Image"])[:,:,None]
            f_rel_image1 = np.array(f_rel1["Image"])[:,:,None]
            f_mask_image = np.concatenate((f_mask_image, f_mask_image1), axis=2)
            f_Direction_image = np.concatenate((f_Direction_image, f_Direction_image1), axis=2)
            f_Inclination_image = np.concatenate((f_Inclination_image, f_Inclination_image1), axis=2)
            f_rel_image = np.concatenate((f_rel_image, f_rel_image1), axis=2)
    return (f_Direction_image, f_Inclination_image, f_mask_image, f_rel_image)



def get_points_noRand(range_r,buffer = 1):
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


def genereate_swirls():
    field_phi = np.empty((60,60,60))
    field_theta = np.copy(field_phi)
    for i in range(field_phi.shape[0]):
        for j in range(field_phi.shape[1]):
            for k in range(field_phi.shape[2]):
                field_phi[i,j,k] = 0 # np.arccos(k+20/np.linalg.norm([i+20,j+20,k+20])) # np.arccos(k+20/np.linalg.norm([i+20,j+20,k+20]))
                field_theta[i,j,k] = np.pi * (i+j)*0.02
    return(field_theta, field_phi)

def get_Y_Odfs_noise(bands:int=10):
    Direction_zero = direction = np.random.uniform(0, 1, (40,40,40))*np.pi
    Inclination_zero = np.zeros((40,40,40))
    mask = np.ones((40,40,40))
    for i in range(Direction_zero.shape[0]):
        for j in range(Direction_zero.shape[1]):
            for k in range(Direction_zero.shape[2]):
                if j <= 25 and i <= 27 and i >= 23 and k <= 50:
                    Direction_zero[i,j,k] = np.pi*(1/2)
                elif i <= 50 and j <= 50 and k <= 50 and j <= -i+52 and j >= -i+48 and j > 25:
                    Direction_zero[i,j,k] = np.pi*(1-1/4)
                elif i <= 50 and j <= 50 and k <= 50 and j <= i+2 and j >= i-2 and j > 25:
                    Direction_zero[i,j,k] = np.pi*(1+1/4)
                # else:
                #     mask[i,j,k] = 0
    _ODFs = odf3.compute(Direction_zero[:,:,:,None], Inclination_zero[:,:,:,None], mask[:,:,:,None], bands)
    return _ODFs    



def kegel_from_dict_withBasis_noTranslation(dict_cache:dict, dict_cache_basis:dict, factor:int=10, alpha:np.ndarray=np.array([]), beta:np.ndarray=np.array([]), get_phi_theta = False):
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

    if get_phi_theta == True:
        # Phi und Theta berechnen für ODFs
        phi = np.arccos(z)
        theta = np.arctan2(y,x)
        return (result, res_basis, phi, theta)

    return result, res_basis

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


def Prepare():
    result, basis, phi, theta = kegel_from_dict_withBasis_noTranslation(dict_res, dict_basis, factor, alpha, beta, True)
    result_rot = reverse_rotate_and_translate_data_noTranslation(result, alpha, beta)
    weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], 0, 0, sigma)
    return result, basis, phi, theta, result_rot, weights



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


def Get_AODF_noRand_noCache_amp(ODFs:np.ndarray, result:np.ndarray, basis:np.ndarray, weights:np.ndarray, phi:np.ndarray, theta:np.ndarray,x_koord:int, y_koord:int, z_koord:int, factor_amp:int = 1000, sigma:float=2,bands:int = 10):
    # Mit weights alle punkte im Kegel ablaufen
    result_ = np.copy(result)
    result_[:,0,:] += x_koord
    result_[:,1,:] += y_koord
    result_[:,2,:] += z_koord
    AODF_Amplitude = get_amplitude(result_, ODFs, basis, weights)

    # # Scatterplot generieren
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
    return (AODF_d, AODF_Amplitude)

## Von https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(samples=1000):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))
    points = np.array(points)
    alpha = np.arctan2(points[:,2],points[:,1])+math.pi/2
    beta = np.arccos(points[:,0])
    return alpha, beta

def fibonacci_sphere_2(num_pts:int=1000):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    theta = np.arccos(1 - 2*indices/num_pts)
    phi = np.pi * (1 + 5**0.5) * indices
    return (phi, theta)

def sphere(number:int=1500):
    phi, theta = fibonacci_sphere_2(number)

    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)


    # fig = plt.figure()
    # ax = fig.add_subplot(projection = "3d")
    # ax.scatter(x, y, z)
    # plt.show()

    points = np.array([x, y, z]).T


    from scipy.spatial import Delaunay, Voronoi
    tri = Delaunay(points)

    # Berechne die äquidistanten Winkel in Kugelkoordinaten
    phi = []
    theta = []

    for simplex in tri.simplices:
        for i in range(3):
            p = points[simplex[i]]
            r, azimuth, inclination = np.linalg.norm(p), np.arctan2(p[1], p[0]), np.arccos(p[2] / np.linalg.norm(p))

            # Füge die Winkel zu den Listen hinzu
            phi.append(azimuth)
            theta.append(inclination)
    return (np.array(phi), np.array(theta))


def get_winkel(num_pts:int=1000):
    phi, theta = fibonacci_sphere(int(num_pts/2))
    phi_, theta_ = fibonacci_sphere_2(int(num_pts/2))
    return  np.append(theta,theta_),np.append(phi,phi_)



def genereate_divergence():
    field_phi = np.empty((40,40,40))
    field_theta = np.copy(field_phi)
    for i in range(field_phi.shape[0]):
        for j in range(field_phi.shape[1]):
            for k in range(field_phi.shape[2]):
                field_phi[i,j,k] = 0 # np.arccos(k+20/np.linalg.norm([i+20,j+20,k+20])) # np.arccos(k+20/np.linalg.norm([i+20,j+20,k+20]))
                field_theta[i,j,k] = np.arctan2(j-20,i-20)
    return(field_theta, field_phi)




def get_Duplicates_new(data=np.array([])):
    _dict_duplicates = dict()
    for i in range(len(data[0,0,:])):
        temp_dict = dict(Counter(map(tuple, np.transpose(np.rint(data[:,:,i].T)))))
        for key, value in temp_dict.items():
            if str(key) == "(nan, nan, nan)":
                continue
            elif _dict_duplicates.get(str(key)) == None:
                _dict_duplicates.update({str(key):value})
            else:
                _dict_duplicates[str(key)] += value
        
    _sorted_dict_duplicates = sorted(dict(_dict_duplicates).items(), key = lambda x:x[1]) 
    return _dict_duplicates, _sorted_dict_duplicates




# def plot_data(data, sorted_data, x_limit, y_limit, z_limit, x_plot_limit = 5, y_plot_limit = 5, z_plot_limit = 5):
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(2,1,1,projection = "3d")

#     red = Color("red")
#     colors = list(red.range_to(Color("blue"), int(list(sorted_data[-1])[-1])+1))

#     # for count, i in enumerate([*dict_duplicates.keys()]):
#     #     if int(list(sorted_dict_duplicates[-1])[-1])*0.5 <= [*dict_duplicates.values()][count]:
#     #         x_str, y_str, z_str_= (i.strip("()").split(","))
#     #         x, y, z = float(x_str), float(y_str), float(z_str_.split(")")[0])
#     #         ax.scatter(x,y,z, c = str(colors[[*dict_duplicates.values()][count]]))


#     for count, i in enumerate([*data.keys()]):
#         x_str, y_str, z_str_= (i.strip("()").split(","))
#         x, y, z = float(x_str), float(y_str), float(z_str_.split(")")[0])
#         if x <= x_limit and y <= y_limit and z <= z_limit:
#             ax.scatter(x,y,z, c = str(colors[[*data.values()][count]]))
#             # if x!=0 and y!=0 and z!=0:
#             #     ax.scatter(x,y,z, c = str(colors[[*data.values()][count]]))

#     #Plot Colormap
#     x_color = range(int(list(sorted_data[-1])[-1])+1)
#     dict_colors = dict(zip(x_color,colors))
#     color_f = lambda n : dict_colors.get(n)
#     # odf3._set_axes_equal(ax)
#     ax1 = fig.add_subplot(2,1,2)
#     # ax1.imshow(aspect='auto')
#     for x_arg in x_color:
#         ax1.axvline(x_arg, color=str(color_f(x_arg)))

#     # pos = ax.get_position()
#     # pos.y0 = 0.8  # Ändere den Startpunkt der Y-Position des Scatterplots
#     # ax.set_position(pos)

#     pos = ax1.get_position()
#     pos.y0 = 0.2  # Ändere den Startpunkt der Y-Position des Scatterplots
#     ax1.set_position(pos)
#     # ax.set_xlim3d(-x_plot_limit, x_plot_limit)
#     # ax.set_ylim3d(-y_plot_limit, y_plot_limit)
#     # ax.set_zlim3d(-z_plot_limit, z_plot_limit)

#     plt.tight_layout()    


def plot_data(data, sorted_data, x_limit, y_limit, z_limit, x_plot_limit = 5, y_plot_limit = 5, z_plot_limit = 5):
    # Erstelle die Figure und die Subplots mit GridSpec
    fig = plt.figure(figsize=(8, 6))
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[8, 1])  # Verhältnis der Höhen der Subplots

    # Erstelle den 3D Scatterplot als oberen Subplot
    ax = plt.subplot(gs[0], projection='3d')


    red = Color("red")
    colors = list(red.range_to(Color("blue"), int(list(sorted_data[-1])[-1])+1))

    # for count, i in enumerate([*dict_duplicates.keys()]):
    #     if int(list(sorted_dict_duplicates[-1])[-1])*0.5 <= [*dict_duplicates.values()][count]:
    #         x_str, y_str, z_str_= (i.strip("()").split(","))
    #         x, y, z = float(x_str), float(y_str), float(z_str_.split(")")[0])
    #         ax.scatter(x,y,z, c = str(colors[[*dict_duplicates.values()][count]]))


    for count, i in enumerate([*data.keys()]):
        x_str, y_str, z_str_= (i.strip("()").split(","))
        x, y, z = float(x_str), float(y_str), float(z_str_.split(")")[0])
        if x <= x_limit and y <= y_limit and z <= z_limit:
            ax.scatter(x,y,z, c = str(colors[[*data.values()][count]]))
            # if x!=0 and y!=0 and z!=0:
            #     ax.scatter(x,y,z, c = str(colors[[*data.values()][count]]))
    # ax.set_xticks([])  # Entferne x-Achsenbeschriftungen
    # ax.set_yticks([])  # Entferne y-Achsenbeschriftungen
    # ax.set_zticks([])
    plt.grid(True)
    #Plot Colormap
    x_color = range(int(list(sorted_data[-1])[-1])+1)
    dict_colors = dict(zip(x_color,colors))
    color_f = lambda n : dict_colors.get(n)
    # odf3._set_axes_equal(ax)

    # Erstelle den Farbverlauf als unteren Subplot
    ax1 = plt.subplot(gs[1])
    # ax1.imshow(aspect='auto')
    for x_arg in x_color:
        ax1.axvline(x_arg, color=str(color_f(x_arg)))


    # Entferne die Beschriftungen der x- und y-Achsen, behalte aber das Gitter
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.get_yaxis().set_ticks([])
    ax.tick_params(axis='x', labelsize=0)
    ax.tick_params(axis='y', labelsize=0)
    ax.tick_params(axis='z', labelsize=0)
    ax1.tick_params(axis='x', labelsize=22)
    # pos = ax.get_position()
    # pos.y0 = 0.8  # Ändere den Startpunkt der Y-Position des Scatterplots
    # ax.set_position(pos)


    # ax.set_xlim3d(-x_plot_limit, x_plot_limit)
    # ax.set_ylim3d(-y_plot_limit, y_plot_limit)
    # ax.set_zlim3d(-z_plot_limit, z_plot_limit)

    plt.tight_layout()    

