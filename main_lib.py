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


def load_dict(range_r = 6, factor = 10):
    with open(f'cache_dict_range_{range_r}_factor_{factor}.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def save_dict(dict_cache={}, range_r = 6, factor = 10):
    with open(f'cache_dict_range_{range_r}_factor_{factor}.pkl', 'wb') as f:
        pickle.dump(dict_cache, f)
    return


def get_array(dict_cache, key):
    return dict_cache[tuple(key)]


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


def get_points(range_r):
    # Punkte vordefinieren
    x = np.arange(-range_r, range_r+1, 1)
    y = np.copy(x)
    z = np.copy(x)
    # erstelle das Gitter für x, y und z
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
    # Punkte außerhalb des Kegels filtern 
    mask = (x_grid**2 + y_grid**2 + z_grid**2 <= (range_r+1)**2) & (x_grid**2 + y_grid**2 + z_grid**2 >= (range_r-1)**2)
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


def generate_dict(range_r = 10, factor = 10):
    Kugelschale, alpha, beta = get_points(factor)
    temp_dict = {tuple(a_row): b_row for a_row, b_row in zip(Kugelschale, koords_in_kegel_cache(range_r, alpha, beta))}
    return temp_dict


def kegel_from_dict(dict_cache={}, factor=10, x_koord=0, y_koord=0, z_koord=0, alpha=np.array([]), beta=np.array([]), paralell_bool=False, get_phi_theta = False):
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
    if paralell_bool == False:
        arrays = [dict_cache[tuple(key)] for key in koords_rounded_int]
        result = np.array(arrays)
    # else:  # langsamer
    #     result = np.array(Parallel(n_jobs=-1)(delayed(get_array)(dict_cache,key) for key in koords_rounded_int))

    result[:,0,:] += x_koord
    result[:,1,:] += y_koord
    result[:,2,:] += z_koord
    if get_phi_theta == True:
        # Phi und Theta berechnen für ODFs
        phi = np.arccos(z)
        theta = np.arctan2(y,x)
        return (result, phi, theta)

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
    for i, (p, ct, st) in enumerate(zip(phi, costheta, sintheta)):
        basis[i, :] = odf3._analytic_single_odf(ct, st, p, band)
    return basis


# def get_basis(phi:np.ndarray, costheta:np.ndarray, sintheta:np.ndarray, band:int)->np.ndarray:
#     basis = np.empty((phi.shape[0], odf.get_num_coeff(band)))
#     for i, (p, ct, st) in enumerate(zip(phi, costheta, sintheta)):
#         basis[i, :] = odf._analytic_single_odf(ct, st, p, band)
#     return basis


def gauss_3d(x,y,z,mu_x=0,mu_y=0,mu_z=0,sigma=2):
    return 1/(np.sqrt(math.pi*2)*sigma)*np.exp(-1/2*(((x-mu_x)/sigma)**2+((y-mu_y)/sigma)**2+((z-mu_z)/sigma)**2))


def gauss_2d(x,y,mu_x=0,mu_y=0,sigma=2):
    return 1/(np.sqrt(math.pi*2)*sigma)*np.exp(-1/2*(((x-mu_x)/sigma)**2+((y-mu_y)/sigma)**2))


def gauss_1d(x,mu_x=0,sigma=2):
    return 1/(np.sqrt(math.pi*2)*sigma)*np.exp(-1/2*(((x-mu_x)/sigma)**2))


def get_basis_xyz(x:int, y:int, z:int, band:int=10):
    phi = np.arccos(z/np.sqrt(x**2+y**2+z**2))
    theta = np.arctan2(y,x)  + np.pi/2
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    basis = np.empty((phi.shape[0], odf3.get_num_coeff(band)))
    for i, (p, ct, st) in enumerate(zip(phi, costheta, sintheta)):
        basis[i, :] = odf3._analytic_single_odf(ct, st, p, band)
    return basis


def generate_dict_basis(range_r:int = 10, factor:int = 10, band: int = 10):
    Kugelschale, alpha, beta = get_points(factor)
    temp_dict = {tuple(a_row): b_row for a_row, b_row in zip(Kugelschale, koords_in_kegel_cache(range_r, alpha, beta))}
    temp_dict_basis = {tuple(a_row): b_row for a_row, b_row in zip(Kugelschale, get_basis_xyz(*Kugelschale.T, band))}
    return temp_dict, temp_dict_basis


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


def get_amplitude(result:np.ndarray, ODFs:np.ndarray, basis:np.ndarray, weights:np.ndarray)->np.ndarray:
    AODF_Amplitude = np.empty((result.shape[0]))
    for i, res in enumerate(result):
        # mask für nan werte
        nan_mask = ~np.isnan(res)
        mask = np.array(res[nan_mask], int)
        # shape anpassen
        mask_3d = np.reshape(mask, (3, int(mask.shape[0]/3)))
        # nur weights auswählen die nicht nan inhalte haben
        weight = weights[i, nan_mask[0]]
        Test_ODF_masked = np.dot(weight[None,:], ODFs[mask_3d[0],mask_3d[1],mask_3d[2]-5])
        sum = np.dot(basis[None,i], Test_ODF_masked.T)
        AODF_Amplitude[i] = sum.item()/res[0,~np.isnan(res[0])].shape[0]
    return AODF_Amplitude


        # phi = np.arccos(z)
        # theta = np.arctan2(y,x)


def get_basis_xyz_new(x:int, y:int, z:int, band:int=10, factor:int=10):
    phi = np.arccos(z/factor)
    theta = np.arctan2(y,x) + np.pi/2
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    basis = np.empty((phi.shape[0], odf3.get_num_coeff(band)))
    for i, (p, ct, st) in enumerate(zip(phi, costheta, sintheta)):
        basis[i, :] = odf3._analytic_single_odf(ct, st, p, band)
    return basis


def generate_dict_basis_new(range_r:int = 10, factor:int = 10, band: int = 10):
    Kugelschale, alpha, beta = get_points(factor)
    temp_dict = {tuple(a_row): b_row for a_row, b_row in zip(Kugelschale, koords_in_kegel_cache(range_r, alpha, beta))}
    temp_dict_basis = {tuple(a_row): b_row for a_row, b_row in zip(Kugelschale, get_basis_xyz_new(*Kugelschale.T, band, factor))}
    return temp_dict, temp_dict_basis