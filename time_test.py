import numpy as np
# Vorherige Imports
import math
import time
import matplotlib.pyplot as plt
from colour import Color
from collections import Counter
import random
from time import gmtime, strftime
from numba import jit, njit
import pickle 
import warnings
from joblib import Parallel, delayed
import timeit



def koords_in_kegel_cache(range_r = 4, alpha = np.array([]), beta = np.array([])):
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    cos_alpha_cos_beta = cos_alpha*cos_beta
    sin_alpha_cos_beta = sin_alpha*cos_beta
    cos_alpha_sin_beta = cos_alpha*sin_beta
    sin_alpha_sin_beta = sin_alpha*sin_beta

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

def load_dict(range_r = 6, factor = 10):
    with open(f'cache_dict_range_{range_r}_factor_{factor}.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def save_dicts(range_r = 6, factor = 10):
    Kugelschale, alpha, beta = get_points(range_r)
    for i in range(2,range_r+1):
        temp_dict = {tuple(a_row): b_row for a_row, b_row in zip(Kugelschale, koords_in_kegel_cache(i, alpha, beta))}
        with open(f'cache_dict_range_{i}_factor_{factor}.pkl', 'wb') as f:
            pickle.dump(temp_dict, f)

def save_dict(dict_cache={}, range_r = 6, factor = 10):
    with open(f'cache_dict_range_{range_r}_factor_{factor}.pkl', 'wb') as f:
        pickle.dump(dict_cache, f)
    return


def get_points(range_r):
    x = np.arange(-range_r, range_r+1, 1)
    y = np.copy(x)
    z = np.copy(x)

    # Erstelle das Gitter für x, y und z
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
    mask = (x_grid**2 + y_grid**2 + z_grid**2 <= (range_r+1)**2) & (x_grid**2 + y_grid**2 + z_grid**2 >= (range_r-1)**2)
    x_mask = x_grid[mask]
    y_mask = y_grid[mask]
    z_mask = z_grid[mask]
    Kugelschale = np.concatenate((x_mask[:,None], y_mask[:,None], z_mask[:,None]), axis=1)
    beta = np.zeros(Kugelschale.shape[0])*np.nan
    mask_1 = (Kugelschale[:,1] >= 0)
    beta[mask_1] = np.arccos(Kugelschale[mask_1,0]/range_r)
    beta[~mask_1] = 2*math.pi - np.arccos(Kugelschale[~mask_1,0]/range_r)
    if np.sum(np.isnan(beta)) > 0:
        print(f"Warning, {np.sum(np.isnan(beta))} nan entries in beta!!!")
    warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    mask_1 = (Kugelschale[:,0] > 0) & (Kugelschale[:,1] >= 0)
    mask_2 = (Kugelschale[:,0] > 0) & (Kugelschale[:,1] < 0)
    mask_3 = (Kugelschale[:,0] < 0)
    # alpha erstallen als Liste mit nur np.nan als inhalt
    alpha = np.zeros(Kugelschale.shape[0])*np.nan
    alpha[mask_1] = np.arctan(-Kugelschale[mask_1,1]/Kugelschale[mask_1,2])
    alpha[mask_2] = np.arctan(-Kugelschale[mask_2,1]/Kugelschale[mask_2,2]) + 2*math.pi
    alpha[mask_3] = np.arctan(-Kugelschale[mask_3,1]/Kugelschale[mask_3,2]) + math.pi
    # Bei beta = pi/2 folgt y = sin(alhpa)*sin(pi/2) = sin(alpha)
    mask_4 = ((beta >= 1.56e+00) & (beta <= 1.58e+00)) 
    mask_5 = ((beta >= 4.70e+00) & (beta <= 4.72e+00))
    mask_final = np.any([mask_4,mask_5], axis=0)
    alpha[mask_final] = np.arcsin(Kugelschale[mask_final,1]/range_r)
    mask_nan = np.isnan(alpha)

    mask_nan = np.isnan(alpha)
    manuell_lst = np.where(mask_nan)[0]

    for i in manuell_lst:
        input_str = math.pi*int(input(f"Bitte geben sie den zu Koordinate: {Kugelschale[i]} und Beta: {beta[i]} passenden Wikel n*\\pi für Alpha an:\n\t"))
        alpha[i] = input_str

    return(Kugelschale, alpha, beta)

def generate_dict(range_r = 10, factor = 10):
    Kugelschale, alpha, beta = get_points(factor)
    temp_dict = {tuple(a_row): b_row for a_row, b_row in zip(Kugelschale, koords_in_kegel_cache(range_r, alpha, beta))}
    return temp_dict


def get_array(dict_cache, key):
    return dict_cache[tuple(key)]


def kegel_from_dict(dict_cache={}, factor=10, x_koord=0, y_koord=0, z_koord=0, alpha=np.array([]), beta=np.array([]), paralell_bool=False):
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
    if paralell_bool == False: # langsamer
        arrays = [dict_cache[tuple(key)] for key in koords_rounded_int]
        result = np.array(arrays)
    else:
        result = np.array(Parallel(n_jobs=-1)(delayed(get_array)(dict_cache,key) for key in koords_rounded_int))

    result[:,0,:] += x_koord
    result[:,1,:] += y_koord
    result[:,2,:] += z_koord
    
    return result


range_r, x_koord, y_koord, z_koord = 5,0,0,0
factor = 50

dict_10_50 = load_dict(10,50)

number_of_winkel = 1000
rng = np.random.default_rng(random.randint(100000,10000000000))
beta = np.arccos(1-2*(rng.random(number_of_winkel).reshape(number_of_winkel,1)))
alpha = rng.random(number_of_winkel).reshape(number_of_winkel,1)*math.pi*2


def foo(x):
    kegel_from_dict(dict_10_50, factor, x_koord, y_koord, z_koord, alpha, beta, False)
    return

foo = lambda x: kegel_from_dict(dict_10_50, factor, x_koord, y_koord, z_koord, alpha, beta, False)

# timeit for x in range(1000): foo(x)
timeit.repeat("for x in range(100): foo(1)")