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

np.set_printoptions(suppress=False, formatter={'float': '{:0.2e}'.format})

def koords_in_kegel_x(range_r = 4):
    # Generiert die Indizes innerhalb eines Kegels entlang der x-Achse mit länge range_r 

    x = np.linspace(-range_r*2, range_r*2, range_r*4+1)
    y = np.copy(x)
    z = np.copy(x)

    # Erstelle das Gitter für x, y und z
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)

    # Mask für innerhalb des Zylinders
    mask = ((z_grid**2 + y_grid**2) < x_grid**2) & (0 < x_grid) & (range_r > x_grid)
    # Filtere Punkte innerhalb des Kegels
    x_mask = x_grid[mask]
    y_mask = y_grid[mask]
    z_mask = z_grid[mask]
    return (x_mask, y_mask, z_mask)


def rotation_kegel(range_r = 10, x_koord = 0, y_koord = 0, z_koord = 0, alpha = np.array([]), beta = np.array([]), no_dup=False):
    # Rotation zuerst um die y-Achse winkel Beta und anschließend um die x-Achse winkel Alpha
    x = np.linspace(-range_r*2, range_r*2, range_r*4+1)
    y = np.copy(x)
    z = np.copy(x)

    # Erstelle das Gitter für x, y und z
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)

    # Mask für innerhalb des Zylinders
    mask = ((z_grid**2 + y_grid**2) < x_grid**2) & (0 < x_grid) & (range_r > x_grid)
    # Filtere Punkte innerhalb des Kegels
    x_mask = x_grid[mask]
    y_mask = y_grid[mask]
    z_mask = z_grid[mask]
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    # Erzeuge 3D-Arrays für x, y und z Koordinaten
    rotation_matrix = np.empty((len(alpha), 3, 3))

    # Fülle die Rotationsmatrizen für alle Winkel gleichzeitig
    rotation_matrix[:, 0, 0] = cos_beta.flatten()
    rotation_matrix[:, 0, 1] = np.zeros_like(cos_beta).flatten()
    rotation_matrix[:, 0, 2] = sin_beta.flatten()

    rotation_matrix[:, 1, 0] = (sin_alpha*sin_beta).flatten()
    rotation_matrix[:, 1, 1] = cos_alpha.flatten()
    rotation_matrix[:, 1, 2] = (-sin_alpha*cos_beta).flatten()

    rotation_matrix[:, 2, 0] = (-cos_alpha*sin_beta).flatten()
    rotation_matrix[:, 2, 1] = sin_alpha.flatten()
    rotation_matrix[:, 2, 2] = (cos_alpha*cos_beta).flatten()
    

    # Dimension der x,y,z anheben
    x_mask_3d = np.tile(x_mask.flatten(), (len(alpha.T), 1))
    y_mask_3d = np.tile(y_mask.flatten(), (len(alpha.T), 1))
    z_mask_3d = np.tile(z_mask.flatten(), (len(alpha.T), 1))

    # Wende die Rotationsmatrix auf die x- und y-Koordinaten an
    result = np.dot(rotation_matrix, np.vstack([x_mask_3d, y_mask_3d, z_mask_3d]))
    # x = result[i,0,:]
    # y = result[i,1,:]
    # z = result[i,2,:]

    result[:,0,:] += x_koord
    result[:,1,:] += y_koord
    result[:,2,:] += z_koord

    # Falls gewollt werden hier die doppelten Punkte die durch das Runden auftreten herausgefilert und mit den Koordinaten 0,0,0 erstetzt
    if no_dup:
        result = np.round(result)
        no_duplicates_with_zero = np.empty(np.shape(result))
        for i in range(np.shape(result)[0]):
            no_duplicates_with_zero[i,:,:] = np.hstack((np.unique(result[i,:,:], axis=1), np.zeros((3, np.shape(result[i,:,:])[1]-np.shape(np.unique(result[i,:,:], axis=1))[1]))))

        return no_duplicates_with_zero
    else:
        return result
    

def get_Duplicates_old(data=np.array([])):
    _dict_duplicates = dict()
    for i in range(len(data[0,0,:])):
        temp_dict = dict(Counter(map(tuple, np.transpose(np.rint(data[:,:,i].T)))))
        for key, value in temp_dict.items():
            if str(key) == "(0.0, 0.0, 0.0)":
                continue
            elif _dict_duplicates.get(str(key)) == None:
                _dict_duplicates.update({str(key):value})
            else:
                _dict_duplicates[str(key)] += value
        
    _sorted_dict_duplicates = sorted(dict(_dict_duplicates).items(), key = lambda x:x[1]) 
    return _dict_duplicates, _sorted_dict_duplicates


def plot_data(data, sorted_data, x_limit, y_limit, z_limit, x_plot_limit = 5, y_plot_limit = 5, z_plot_limit = 5):
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")

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

    #Plot Colormap
    x_color = range(int(list(sorted_data[-1])[-1])+1)
    dict_colors = dict(zip(x_color,colors))
    color_f = lambda n : dict_colors.get(n)

    fig1, ax1 = plt.subplots(figsize=(6, 1))

    for x_arg in x_color:
        ax1.axvline(x_arg, color=str(color_f(x_arg)))

    # ax.set_xlim3d(-x_plot_limit, x_plot_limit)
    # ax.set_ylim3d(-y_plot_limit, y_plot_limit)
    # ax.set_zlim3d(-z_plot_limit, z_plot_limit)

    plt.tight_layout()    
    plt.show()


def get_Duplicates(data):
    _dict_duplicates = {}
    for lst_for in data:
        if _dict_duplicates.get(str(lst_for)) == None:
            _dict_duplicates.update({str(lst_for):1})
        else:
            _dict_duplicates[str(lst_for)] += 1
    return _dict_duplicates


# def reverse_rotate_and_translate_data(data, alpha,  beta, x_delta=0, y_delta=0, z_delta=0):
#     sin_alpha = np.sin(-alpha)
#     sin_beta = np.sin(-beta)
#     cos_alpha = np.cos(-alpha)
#     cos_beta = np.cos(-beta)

#     x,y,z = data
#     x -= x_delta
#     y -= y_delta
#     z -= z_delta
#     # erst rotation um -alpha um die x achse, dann rotation um -beta um die y achse
#     x_rot = z*cos_alpha*sin_beta + x*cos_beta + y*sin_alpha*sin_beta
#     y_rot = y*cos_alpha - z*sin_alpha
#     z_rot = y*sin_alpha*cos_beta - x*sin_beta + z*cos_alpha*cos_beta

#     # erst rotation um beta um die y achse, dann rotation um alpha um die x achse
#     # x_rot = z*sin_beta + x*cos_beta
#     # y_rot = y*cos_alpha + x*sin_alpha*sin_beta - z*sin_alpha*cos_beta
#     # z_rot = y*sin_alpha - x*cos_alpha*sin_beta + z*cos_alpha*cos_beta

#     return(x_rot, y_rot, z_rot)


def gauss_3d(x,y,z,mu_x=0,mu_y=0,mu_z=0,sigma=2):
    return 1/(np.sqrt(math.pi*2)*sigma)*np.exp(-1/2*(((x-mu_x)/sigma)**2+((y-mu_y)/sigma)**2+((z-mu_z)/sigma)**2))


def gauss_2d(x,y,mu_x=0,mu_y=0,sigma=2):
    return 1/(np.sqrt(math.pi*2)*sigma)*np.exp(-1/2*(((x-mu_x)/sigma)**2+((y-mu_y)/sigma)**2))


def gauss_1d(x,mu_x=0,sigma=2):
    return 1/(np.sqrt(math.pi*2)*sigma)*np.exp(-1/2*(((x-mu_x)/sigma)**2))




# from numba import jit, int32


# def get_mesh(range_r = 4, x_koord = 0, y_koord = 0, z_koord = 0, alpha = np.array([]), beta = np.array([])):
#     cos_alpha = np.cos(alpha)
#     sin_alpha = np.sin(alpha)
#     cos_beta = np.cos(beta)
#     sin_beta = np.sin(beta)

#     x = np.linspace(-range_r*2, range_r*2, range_r*4+1)
#     y = np.copy(x)
#     z = np.copy(x)
    
#     # Erstelle das Gitter für x, y und z
#     x_grid, y_grid, z_grid = np.meshgrid(x, y, z)

#     return loop_winkel(range_r, x_koord, y_koord, z_koord, cos_alpha, sin_alpha, cos_beta, sin_beta, x_grid, y_grid, z_grid )

# @jit(nopython=True)
# def loop_winkel(range_r = 4, x_koord = 0, y_koord = 0, z_koord = 0, cos_alpha = np.array([]), sin_alpha = np.array([]), cos_beta = np.array([]), sin_beta = np.array([]), x_grid = np.array([]), y_grid = np.array([]), z_grid = np.array([])):
#     # Generiert die Indizes innerhalb eines Kegels entlang der x-Achse mit länge range_r 
#     for i ,ca, sa, cb, sb in zip(range(len(cos_alpha)), cos_alpha, sin_alpha, cos_beta, sin_beta):
#         mask = (((y_grid*sa - x_grid*ca*sb + z_grid*ca*cb)**2 + (y_grid*ca + x_grid*sa*sb - z_grid*sa*cb)**2) < (z_grid*sb + x_grid*cb)**2) & (0 < (z_grid*sb + x_grid*cb)) & (range_r > (z_grid*sb + x_grid*cb))
#         x_mask = x_grid[mask] + x_koord
#         y_mask = y_grid[mask] + y_koord
#         z_mask = z_grid[mask] + z_koord
#         shape = np.shape((x_mask, y_mask, z_mask)) 
#         if i == 0:
#             max_lenght = shape[-1]*2
#             n = len(cos_alpha)
#             result = np.empty((n, 3, max_lenght))
#             result[:] = np.nan   
#             # print(np.shape(result))
#             # print(np.shape((x_mask, y_mask, z_mask)))
#             result[i, :, :shape[1]] = (x_mask, y_mask, z_mask)
#         else:
#             # print(np.shape(result))
#             # print(np.shape((x_mask, y_mask, z_mask)))
#             result[i, :, :shape[1]] = (x_mask, y_mask, z_mask)

#     return result



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



def kegel_meshgrid(range_r = 4, x_koord = 0, y_koord = 0, z_koord = 0, alpha = np.array([]), beta = np.array([])):
# Generiert die Indizes innerhalb eines Kegels entlang der x-Achse mit länge range_r 
    # print(cos_alpha,sin_alpha,cos_beta,sin_beta)
    x = np.linspace(-range_r*2, range_r*2, range_r*4+1)

    # Erstelle das Gitter für x, y und z
    x_grid, y_grid, z_grid = np.meshgrid(x, x, x)
    # test_x = np.copy(x_grid)
    x_mask, y_mask, z_mask = where_func(x_grid, y_grid, z_grid, range_r, alpha, beta, x_koord, y_koord, z_koord)
    return combnine(x_mask, y_mask, z_mask)


@jit(nopython=True, cache=True)
def where_func(x_grid=np.array([[],[],[]]), y_grid=np.array([[],[],[]]), z_grid=np.array([[],[],[]]), range_r=5, alpha=np.array([]), beta=np.array([]),x_koord=0, y_koord=0, z_koord=0):  
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    x_grid_tiled = x_grid.flatten().repeat(len(alpha)).reshape((-1, len(alpha))).T
    y_grid_tiled = y_grid.flatten().repeat(len(alpha)).reshape((-1, len(alpha))).T
    z_grid_tiled = z_grid.flatten().repeat(len(alpha)).reshape((-1, len(alpha))).T
    mask_3d = (((cos_alpha*y_grid_tiled-sin_alpha*z_grid_tiled)**2 + (cos_alpha*cos_beta*z_grid_tiled-sin_beta*x_grid_tiled+sin_alpha*cos_beta*y_grid_tiled)**2 < (cos_alpha*sin_beta*z_grid_tiled+sin_alpha*sin_beta*y_grid_tiled+cos_beta*x_grid_tiled)**2) & (0 < (cos_alpha*sin_beta*z_grid_tiled+sin_alpha*sin_beta*y_grid_tiled+cos_beta*x_grid_tiled)) & (range_r > (cos_alpha*sin_beta*z_grid_tiled+sin_alpha*sin_beta*y_grid_tiled+cos_beta*x_grid_tiled)))
    x_mask = np.where(mask_3d, x_grid_tiled, np.nan) + x_koord
    y_mask = np.where(mask_3d, y_grid_tiled, np.nan) + y_koord
    z_mask = np.where(mask_3d, z_grid_tiled, np.nan) + z_koord

    # result = np.array([x_mask+x_koord, y_mask+y_koord, z_mask+z_koord])
    # result = np.swapaxes(result, 0,1)
    # return result
    return (x_mask, y_mask, z_mask)

def combnine(x_mask, y_mask, z_mask):
    result = np.array([x_mask, y_mask, z_mask])
    result = np.swapaxes(result, 0,1)
    return result



def kegel_meshgrid_repeat(range_r = 4, x_koord = 0, y_koord = 0, z_koord = 0, alpha = np.array([]), beta = np.array([])):
# Generiert die Indizes innerhalb eines Kegels entlang der x-Achse mit länge range_r 
    # print(cos_alpha,sin_alpha,cos_beta,sin_beta)
    x = np.linspace(-int(np.ceil(range_r*1.42)), int(np.ceil(range_r*1.42)), int(np.ceil(range_r*1.42)*2+1))

    # Erstelle das Gitter für x, y und z
    x_grid, y_grid, z_grid = np.meshgrid(x, x, x)
    # test_x = np.copy(x_grid)
    mask = where_func_repeat(x_grid, y_grid, z_grid, range_r, alpha, beta, x_koord, y_koord, z_koord)
    return combnine_repeat(x_grid,y_grid,z_grid,mask)


@njit(cache=True)
def where_func_repeat(x_grid=np.array([[],[],[]]), y_grid=np.array([[],[],[]]), z_grid=np.array([[],[],[]]), range_r=5, alpha=np.array([]), beta=np.array([]),x_koord=0, y_koord=0, z_koord=0):  
    cos_alpha = np.cos(alpha).T
    sin_alpha = np.sin(alpha).T
    cos_beta = np.cos(beta).T
    sin_beta = np.sin(beta).T
    x_grid_repeat = x_grid.repeat(len(alpha)).reshape(*np.shape(x_grid),len(alpha))
    y_grid_repeat = y_grid.repeat(len(alpha)).reshape(*np.shape(x_grid),len(alpha))
    z_grid_repeat = z_grid.repeat(len(alpha)).reshape(*np.shape(x_grid),len(alpha))

    x_rot = (np.multiply(cos_alpha*sin_beta,z_grid_repeat)+np.multiply(sin_alpha*sin_beta,y_grid_repeat)+np.multiply(cos_beta,x_grid_repeat))
    y_rot = (np.multiply(y_grid_repeat,cos_alpha)-np.multiply(sin_alpha,z_grid_repeat))
    z_rot = (np.multiply(cos_alpha*cos_beta,z_grid_repeat)-np.multiply(sin_beta,x_grid_repeat)+np.multiply(sin_alpha*cos_beta,y_grid_repeat))

    mask = (y_rot**2+z_rot**2 < x_rot**2) & (x_rot < range_r) & (x_rot > 0)
    return (mask)

def combnine_repeat(x_grid,y_grid,z_grid,mask):
    res = np.ones((np.shape(mask)[-1], 3, int(np.shape(x_grid)[0]**3/2+1)))*np.nan
    for i in range(np.shape(mask)[-1]):
        res[i,0,:np.shape(x_grid[mask[:,:,:,i]])[-1]] = x_grid[mask[:,:,:,i]]
        res[i,1,:np.shape(y_grid[mask[:,:,:,i]])[-1]] = y_grid[mask[:,:,:,i]]
        res[i,2,:np.shape(z_grid[mask[:,:,:,i]])[-1]] = z_grid[mask[:,:,:,i]]
    return res



def koords_in_kegel_alpha_beta_test(range_r = 4, x_koord = 0, y_koord = 0, z_koord = 0, alpha = np.array([]), beta = np.array([])):
# Generiert die Indizes innerhalb eines Kegels entlang der x-Achse mit länge range_r 
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    # print(cos_alpha,sin_alpha,cos_beta,sin_beta)
    x = np.linspace(-range_r*2, range_r*2, range_r*4+1)
    y = np.copy(x)
    z = np.copy(x)
    length = int(np.ceil((2*range_r)**3/6))+1
    
    # Erstelle das Gitter für x, y und z
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
    # test_x = np.copy(x_grid)
    result = np.empty((len(cos_alpha), 3, length)) * np.nan
    for i, ca, sa, cb, sb in zip(range(len(cos_alpha)), cos_alpha, sin_alpha, cos_beta, sin_beta):
        # print(cos_alpha[i], sin_alpha[i], cos_beta[i], sin_beta[i])
        # print(ca,sa,cb,sb)
        # erst x achse dann y achse
        mask = (((ca*y_grid-sa*z_grid)**2 + (ca*cb*z_grid-sb*x_grid+sa*cb*y_grid)**2 < (ca*sb*z_grid+sa*sb*y_grid+cb*x_grid)**2) & (0 < (ca*sb*z_grid+sa*sb*y_grid+cb*x_grid)) & (range_r > (ca*sb*z_grid+sa*sb*y_grid+cb*x_grid)))
        # x,y,z werte mit der spezifischen Maske auf die Kegel zuschneiden
        x_mask = x_grid[mask]
        y_mask = y_grid[mask]
        z_mask = z_grid[mask]
        data = np.array([x_mask + x_koord, y_mask + y_koord, z_mask + z_koord])
        result[i,:,:np.shape(data)[1]] = data
    return result


def kegel_meshgrid_where(range_r = 4, x_koord = 0, y_koord = 0, z_koord = 0, alpha = np.array([]), beta = np.array([])):
# Generiert die Indizes innerhalb eines Kegels entlang der x-Achse mit länge range_r 
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    # print(cos_alpha,sin_alpha,cos_beta,sin_beta)
    x = np.linspace(-range_r*2, range_r*2, range_r*4+1)
    # y = np.copy(x)
    # z = np.copy(x)
    
    # Erstelle das Gitter für x, y und z
    x_grid, y_grid, z_grid = np.meshgrid(x, x, x)
    # test_x = np.copy(x_grid)
    x_grid_tiled = np.tile(x_grid.flatten(), (len(alpha), 1))
    y_grid_tiled = np.tile(y_grid.flatten(), (len(alpha), 1))
    z_grid_tiled = np.tile(z_grid.flatten(), (len(alpha), 1))

    mask_3d = (((cos_alpha*y_grid_tiled-sin_alpha*z_grid_tiled)**2 + (cos_alpha*cos_beta*z_grid_tiled-sin_beta*x_grid_tiled+sin_alpha*cos_beta*y_grid_tiled)**2 < (cos_alpha*sin_beta*z_grid_tiled+sin_alpha*sin_beta*y_grid_tiled+cos_beta*x_grid_tiled)**2) & (0 < (cos_alpha*sin_beta*z_grid_tiled+sin_alpha*sin_beta*y_grid_tiled+cos_beta*x_grid_tiled)) & (range_r > (cos_alpha*sin_beta*z_grid_tiled+sin_alpha*sin_beta*y_grid_tiled+cos_beta*x_grid_tiled)))
    x_mask = np.where(mask_3d, x_grid_tiled, np.nan)
    y_mask = np.where(mask_3d, y_grid_tiled, np.nan)
    z_mask = np.where(mask_3d, z_grid_tiled, np.nan)

    result = np.swapaxes(np.array([x_mask+x_koord, y_mask+y_koord, z_mask+z_koord]), 0,1)
    return result


def Kegel_Hybrid(x_koord = 0, y_koord = 0, z_koord = 0, range_r = 4, alpha = np.array([]), beta = np.array([]), no_dup=False):
    # Rotation zuerst um die y-Achse winkel Beta und anschließend um die x-Achse winkel Alpha

    x = np.linspace(-range_r*2, range_r*2, range_r*4+1)
    y = np.copy(x)
    z = np.copy(x)

    # Erstelle das Gitter für x, y und z
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)

    # Mask für innerhalb des Zylinders
    mask = ((z_grid**2 + y_grid**2) < x_grid**2) & (0 < x_grid) & (range_r > x_grid)
    # Filtere Punkte innerhalb des Kegels
    x_mask = x_grid[mask]
    y_mask = y_grid[mask]
    z_mask = z_grid[mask]

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    # Erzeuge 3D-Arrays für x, y und z Koordinaten
    rotation_matrix = np.empty((len(alpha), 3, 3))

    # Fülle die Rotationsmatrizen für alle Winkel gleichzeitig
    rotation_matrix[:, 0, 0] = cos_beta.ravel()
    rotation_matrix[:, 0, 1] = np.zeros_like(cos_beta).ravel()
    rotation_matrix[:, 0, 2] = sin_beta.ravel()

    rotation_matrix[:, 1, 0] = (sin_alpha*sin_beta).ravel()
    rotation_matrix[:, 1, 1] = cos_alpha.ravel()
    rotation_matrix[:, 1, 2] = (-sin_alpha*cos_beta).ravel()

    rotation_matrix[:, 2, 0] = (-cos_alpha*sin_beta).ravel()
    rotation_matrix[:, 2, 1] = sin_alpha.ravel()
    rotation_matrix[:, 2, 2] = (cos_alpha*cos_beta).ravel()
    

    # Dimension der x,y,z anheben
    x_mask_3d = np.tile(x_mask.ravel(), (len(alpha.T), 1))
    y_mask_3d = np.tile(y_mask.ravel(), (len(alpha.T), 1))
    z_mask_3d = np.tile(z_mask.ravel(), (len(alpha.T), 1))

    # Wende die Rotationsmatrix auf die x- und y-Koordinaten an
    result = np.dot(rotation_matrix, np.vstack([x_mask_3d, y_mask_3d, z_mask_3d]))
    # x = result[:,0,:]
    # y = result[:,1,:]
    # z = result[:,2,:]



    result[:,0,:] += x_koord
    result[:,1,:] += y_koord
    result[:,2,:] += z_koord


    # Falls gewollt werden hier die doppelten Punkte die durch das Runden auftreten herausgefilert und mit den Koordinaten 0,0,0 erstetzt
    return result


from numba import njit, int32
from Test_lib import *



def get_limits(range_r = 4, alpha = np.array([]), beta = np.array([]), R_Faktor = 0.7071067812):
    # Rotation zuerst um die y-Achse winkel Beta und anschließend um die x-Achse winkel Alpha
    # # Filtere Punkte innerhalb des Kegels
    x_mask = np.array([range_r, range_r])
    y_mask = np.array([R_Faktor*range_r, -R_Faktor*range_r])
    z_mask = np.array([R_Faktor*range_r, -R_Faktor*range_r])

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    # Erzeuge 3D-Arrays für x, y und z Koordinaten
    rotation_matrix = np.empty((len(alpha), 3, 3))

    # Fülle die Rotationsmatrizen für alle Winkel gleichzeitig
    rotation_matrix[:, 0, 0] = cos_beta.ravel()
    rotation_matrix[:, 0, 1] = np.zeros_like(cos_beta).ravel()
    rotation_matrix[:, 0, 2] = sin_beta.ravel()

    rotation_matrix[:, 1, 0] = (sin_alpha*sin_beta).ravel()
    rotation_matrix[:, 1, 1] = cos_alpha.ravel()
    rotation_matrix[:, 1, 2] = (-sin_alpha*cos_beta).ravel()

    rotation_matrix[:, 2, 0] = (-cos_alpha*sin_beta).ravel()
    rotation_matrix[:, 2, 1] = sin_alpha.ravel()
    rotation_matrix[:, 2, 2] = (cos_alpha*cos_beta).ravel()
    

    # Dimension der x,y,z anheben
    x_mask_3d = np.tile(x_mask.ravel(), (len(alpha.T), 1))
    y_mask_3d = np.tile(y_mask.ravel(), (len(alpha.T), 1))
    z_mask_3d = np.tile(z_mask.ravel(), (len(alpha.T), 1))

    # Wende die Rotationsmatrix auf die x- und y-Koordinaten an
    result = np.dot(rotation_matrix, np.vstack([x_mask_3d, y_mask_3d, z_mask_3d]))
    # x = result[:,0,:]
    # y = result[:,1,:]
    # z = result[:,2,:]

    # Falls gewollt werden hier die doppelten Punkte die durch das Runden auftreten herausgefilert und mit den Koordinaten 0,0,0 erstetzt
    return result



def mesh_loop(range_r=7, Offset_i=12, Offset_j=12, Offset_k=12, alpha=np.array([]), beta=np.array([])):
    # Länge des Kegels
    # limit_r = 10
    # Offset des Kegels in x-Richtung
    # Offset_i, Offset_j, Offset_k = 12,12,12
    # alpha: drehwinkel um die j/x-Achse, beta: drehwinkel um die k/z-Achse
    limits = get_limits(range_r, alpha, beta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    # print((sin_beta.shape[-1],*np.array(koords_in_kegel_x(range_r)).shape))
    shape = np.array(koords_in_kegel_x(range_r)).shape
    result = np.ones((sin_beta.shape[0],shape[0],shape[1]*2))*np.nan
    # print(result.shape)
    # result = np.ones((alpha.shape[0], ))
    # for i in np.arange((np.min(limit[0]) < 0) * np.floor(np.min(limit[0])), (np.max(limit[0]) > 0) * np.ceil(np.max(limit[0]))) 
    # for j in np.arange(np.floor(np.min(limit[1])), np.ceil(np.max(limit[1]))) 
    # for k in np.arange(np.floor(np.min(limit[2])), np.ceil(np.max(limit[2]))) 
    for i, ca, sa, cb, sb, limit in zip(range(cos_alpha.shape[0]), cos_alpha, sin_alpha, cos_beta, sin_beta, limits):
        # data =[[i+Offset_i,j+Offset_j,k+Offset_k]  
        #     for i in np.arange(-np.ceil(np.max(np.abs(limit[0])))-1, np.ceil(np.max(np.abs(limit[0])))+1) 
        #     for j in np.arange(-np.ceil(np.max(np.abs(limit[1])))-1, np.ceil(np.max(np.abs(limit[1])))+1) 
        #     for k in np.arange(-np.ceil(np.max(np.abs(limit[2])))-1, np.ceil(np.max(np.abs(limit[2])))+1) 
        #     if (0 < (ca*sb*k+sa*sb*j+cb*i))
        #     if ((ca*j-sa*k)**2 + (ca*cb*k-sb*i+sa*cb*j)**2 < (ca*sb*k+sa*sb*j+cb*i)**2)
        #     if (range_r > (ca*sb*k+sa*sb*j+cb*i))
        #     ] 
        x = np.arange(-np.ceil(np.max(np.abs(limit[0])))-1, np.ceil(np.max(np.abs(limit[0])))+1) 
        y = np.arange(-np.ceil(np.max(np.abs(limit[1])))-1, np.ceil(np.max(np.abs(limit[1])))+1)
        z = np.arange(-np.ceil(np.max(np.abs(limit[2])))-1, np.ceil(np.max(np.abs(limit[2])))+1)
        
        # x = np.arange(-np.ceil(np.max(np.abs(limit[0])))-1, np.ceil(np.max(np.abs(limit[0])))+1) 
        # y = np.arange(np.floor(np.min(limit[1])), np.ceil(np.max(limit[1])))
        # z = np.arange(np.floor(np.min(limit[2])), np.ceil(np.max(limit[2])))

        # Erstelle das Gitter für x, y und z
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
        print(x_grid.shape)
        mask = (((ca*y_grid-sa*z_grid)**2 + (ca*cb*z_grid-sb*x_grid+sa*cb*y_grid)**2 < (ca*sb*z_grid+sa*sb*y_grid+cb*x_grid)**2) & (0 < (ca*sb*z_grid+sa*sb*y_grid+cb*x_grid)) & (range_r > (ca*sb*z_grid+sa*sb*y_grid+cb*x_grid)))
        x_mask = x_grid[mask]
        y_mask = y_grid[mask]
        z_mask = z_grid[mask]
        # print(x_mask.shape)
        data = np.array([x_mask + Offset_i, y_mask + Offset_j, z_mask + Offset_k])
        result[i,:,:x_mask.shape[0]] = data

        # result[i,0,:x_mask.shape[0]] = x_mask + Offset_i
        # result[i,1,:x_mask.shape[0]] = y_mask + Offset_j
        # result[i,2,:x_mask.shape[0]] = z_mask + Offset_k
        
    return result



##################### Chache

def koords_in_kegel_cache(range_r = 4, alpha = np.array([]), beta = np.array([])):
    cos_alpha = np.cos(-alpha)
    sin_alpha = np.sin(-alpha)
    cos_beta = np.cos(-beta)
    sin_beta = np.sin(-beta)
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




def Chache_Check(chache_aplha_beta, alpha, beta, round_int=3):
    in_question = np.concatenate((alpha, beta), axis=1)
    mask = np.isin(np.round(in_question,round_int),np.round(chache_aplha_beta,round_int))
    mask_results = mask[:,0] * mask[:,1]
    not_in_cache = in_question[mask_results,:]
    in_cache = in_question[~mask_results,:]

    return (not_in_cache[:,0], not_in_cache[:,1]) , (in_cache[:,0], in_cache[:,1])


def fill_Cache(Cache_index=None, Cache_value=None, alpha=None, beta=None, result=None, round_int=3):
    if Cache_index is None:
        # Erstellen des Caches mit den ersten Einträgen
        raw_alpha_beta = np.round(np.concatenate((alpha, beta), axis=1), round_int)
        # Cache Runden damit gleiche winkel auftreten können
        mask = np.unique(raw_alpha_beta, True , axis=0)[1]
        Cache_value = result[mask,:,:]
        Cache_index = raw_alpha_beta[mask]
        # Sortieren des Caches 
        mask_sort = np.argsort(Cache_index[:,0])
        Cache_index_sorted = Cache_index[mask_sort,:]
        Cache_value_sorted = Cache_value[mask_sort,:,:]

    else:
        raw_alpha_beta = np.round(np.concatenate((alpha, beta), axis=1), round_int)
        mask = np.unique(raw_alpha_beta, True , axis=0)[1]
        pot_index = raw_alpha_beta[mask]
        pot_val = result[mask,:,:]
        # mask_dup = np.isin(pot_index[:,0], Cache_index[:,0], invert=True) * np.isin(pot_index[:,1], Cache_index[:,1], invert=True)
        mask_dup_ges = np.isin(pot_index, Cache_index, invert=True)
        mask_dup = mask_dup_ges[:,0] * mask_dup_ges[:,1]
        new_index = pot_index[mask_dup]
        print(new_index.shape)
        new_val = pot_val[mask_dup,:,:]
        mask_sort = np.argsort(new_index[:,0])
        new_index_sorted = new_index[mask_sort,:]
        new_val_sorted = new_val[mask_sort,:,:]
        new_cache_index = np.concatenate((Cache_index, new_index_sorted), axis=0)
        new_cache_value = np.concatenate((Cache_value, new_val_sorted), axis=0)
        mask_sort_final = np.argsort(new_cache_index[:,0])
        Cache_index_sorted = new_cache_index[mask_sort_final,:]
        Cache_value_sorted = new_cache_value[mask_sort_final,:,:]

    return Cache_index_sorted, Cache_value_sorted         




def fill_Cache_isClose(Cache_index=None, Cache_value=None, alpha=None, beta=None, result=None, tolerance=3):
    # tolerance = float(f"1e-0{close_int}")
    if Cache_index is None:
        # Erstellen des Caches mit den ersten Einträgen
        raw_alpha_beta = np.concatenate((alpha, beta), axis=1)
        diffs = np.linalg.norm(np.diff(raw_alpha_beta, axis=0), axis=1)
        mask_dup = np.where(diffs<tolerance)[0]
        np.delete(raw_alpha_beta, mask_dup+1, axis=0)
        mask_sort = np.argsort(raw_alpha_beta[:,0])
        Cache_index_sorted = raw_alpha_beta[mask_sort,:]
        Cache_value_sorted = result[mask_sort,:,:]

    else:
        raw_alpha_beta = np.concatenate((alpha, beta), axis=1)
        # mask = np.unique(raw_alpha_beta, True , axis=0)[1]
        # pot_index = raw_alpha_beta[mask]
        # pot_val = result[mask,:,:]
        # mask_dup = np.isin(pot_index[:,0], Cache_index[:,0], invert=True) * np.isin(pot_index[:,1], Cache_index[:,1], invert=True)
        mask_dup = np.all(np.isclose(raw_alpha_beta[:, None], Cache_index, rtol=tolerance), axis=-1).any(-1)
        # inverted_mask_dup = np.ones(raw_alpha_beta.shape[0] ,dtype=bool)
        # inverted_mask_dup[mask_dup] = False
        # new_index = raw_alpha_beta[inverted_mask_dup]
        # new_val = result[inverted_mask_dup,:,:]
        new_index = raw_alpha_beta[~mask_dup]
        new_val = result[~mask_dup,:,:]
        mask_sort = np.argsort(new_index[:,0])
        new_index_sorted = new_index[mask_sort,:]
        new_val_sorted = new_val[mask_sort,:,:]
        new_cache_index = np.concatenate((Cache_index, new_index_sorted), axis=0)
        new_cache_value = np.concatenate((Cache_value, new_val_sorted), axis=0)
        mask_sort_final = np.argsort(new_cache_index[:,0])
        Cache_index_sorted = new_cache_index[mask_sort_final,:]
        Cache_value_sorted = new_cache_value[mask_sort_final,:,:]

    return Cache_index_sorted, Cache_value_sorted     




def fill_Cache_round(Cache_index=None, Cache_value=None, alpha=None, beta=None, result=None, round_int=3):
    if Cache_index is None:
        # Erstellen des Caches mit den ersten Einträgen
        raw_alpha_beta = np.round(np.concatenate((alpha, beta), axis=1), round_int)
        # Cache Runden damit gleiche winkel auftreten können
        # Doppelte herausfiltern, [0] ist die liste der unique
        Cache_index, mask = np.unique(raw_alpha_beta, True , axis=0)
        Cache_value = result[mask,:,:]
        # Sortieren des Caches 
        mask_sort = np.argsort(Cache_index[:,0])
        Cache_index_sorted = Cache_index[mask_sort,:]
        Cache_value_sorted = Cache_value[mask_sort,:,:]

    else:
        raw_alpha_beta = np.round(np.concatenate((alpha, beta), axis=1), round_int)
        pot_index, mask = np.unique(raw_alpha_beta, True , axis=0)
        pot_val = result[mask,:,:]
        # mask_dup = np.isin(pot_index[:,0], Cache_index[:,0], invert=True) * np.isin(pot_index[:,1], Cache_index[:,1], invert=True)
        mask_dup = np.isin(pot_index, Cache_index, invert=True)
        new_index = pot_index[mask_dup]
        print(new_index.shape)
        new_val = pot_val[mask_dup,:,:]
        mask_sort = np.argsort(new_index[:,0])
        new_index_sorted = new_index[mask_sort,:]
        new_val_sorted = new_val[mask_sort,:,:]
        new_cache_index = np.concatenate((Cache_index, new_index_sorted), axis=0)
        new_cache_value = np.concatenate((Cache_value, new_val_sorted), axis=0)
        mask_sort_final = np.argsort(new_cache_index[:,0])
        Cache_index_sorted = new_cache_index[mask_sort_final,:]
        Cache_value_sorted = new_cache_value[mask_sort_final,:,:]

    return np.array(Cache_index_sorted), np.array(Cache_value_sorted)    



def array_row_intersection_mask(a,b):
   tmp=np.prod(np.swapaxes(a[:,:,None], 1, 2) == b, axis = 2)
   return np.sum(np.cumsum(tmp, axis = 0) * tmp == 1, axis = 1).astype(bool)



def fill_Cache_round_intersect(Cache_index=None, Cache_value=None, alpha=None, beta=None, result=None, round_int=3):
    if Cache_index is None:
        # Erstellen des Caches mit den ersten Einträgen
        raw_alpha_beta = np.round(np.concatenate((alpha, beta), axis=1), round_int)
        # Cache Runden damit gleiche winkel auftreten können
        # Doppelte herausfiltern, [0] ist die liste der unique
        Cache_index, mask = np.unique(raw_alpha_beta, True , axis=0)
        Cache_value = result[mask,:,:]
        # Sortieren des Caches 
        mask_sort = np.argsort(Cache_index[:,0])
        Cache_index_sorted = Cache_index[mask_sort,:]
        Cache_value_sorted = Cache_value[mask_sort,:,:]

    else:
        raw_alpha_beta = np.round(np.concatenate((alpha, beta), axis=1), round_int)
        pot_index, mask = np.unique(raw_alpha_beta, True , axis=0)
        pot_val = result[mask,:,:]
        # mask_dup = np.isin(pot_index[:,0], Cache_index[:,0], invert=True) * np.isin(pot_index[:,1], Cache_index[:,1], invert=True)
        mask_dup = array_row_intersection_mask(pot_index, Cache_index)
        new_index = pot_index[~mask_dup]
        print(new_index.shape)
        new_val = pot_val[~mask_dup,:,:]
        mask_sort = np.argsort(new_index[:,0])
        new_index_sorted = new_index[mask_sort,:]
        new_val_sorted = new_val[mask_sort,:,:]
        new_cache_index = np.concatenate((Cache_index, new_index_sorted), axis=0)
        new_cache_value = np.concatenate((Cache_value, new_val_sorted), axis=0)
        mask_sort_final = np.argsort(new_cache_index[:,0])
        Cache_index_sorted = new_cache_index[mask_sort_final,:]
        Cache_value_sorted = new_cache_value[mask_sort_final,:,:]

    return np.array(Cache_index_sorted), np.array(Cache_value_sorted)    



def fill_Cache_round_bracket(Cache_index=None, Cache_value=None, alpha=None, beta=None, result=None, round_int=3):
    if Cache_index is None:
        # Erstellen des Caches mit den ersten Einträgen
        raw_alpha_beta = np.round(np.concatenate((alpha, beta), axis=1), round_int)
        # Cache Runden damit gleiche winkel auftreten können
        # Doppelte herausfiltern, [0] ist die liste der unique
        Cache_index, mask = np.unique(raw_alpha_beta, True , axis=0)
        Cache_value = result[mask,:,:]
        # Sortieren des Caches 
        mask_sort = np.argsort(Cache_index[:,0])
        Cache_index_sorted = Cache_index[mask_sort,:]
        Cache_value_sorted = Cache_value[mask_sort,:,:]

    else:
        raw_alpha_beta = np.round(np.concatenate((alpha, beta), axis=1), round_int)
        pot_index, mask = np.unique(raw_alpha_beta, True , axis=0)
        pot_val = result[mask,:,:]
        # mask_dup = np.isin(pot_index[:,0], Cache_index[:,0], invert=True) * np.isin(pot_index[:,1], Cache_index[:,1], invert=True)
        mask_dup = (pot_index[:, None] == Cache_index).all(-1).any(1)
        new_index = pot_index[~mask_dup]
        print(new_index.shape)
        new_val = pot_val[~mask_dup,:,:]
        mask_sort = np.argsort(new_index[:,0])
        new_index_sorted = new_index[mask_sort,:]
        new_val_sorted = new_val[mask_sort,:,:]
        new_cache_index = np.concatenate((Cache_index, new_index_sorted), axis=0)
        new_cache_value = np.concatenate((Cache_value, new_val_sorted), axis=0)
        mask_sort_final = np.argsort(new_cache_index[:,0])
        Cache_index_sorted = new_cache_index[mask_sort_final,:]
        Cache_value_sorted = new_cache_value[mask_sort_final,:,:]

    return np.array(Cache_index_sorted), np.array(Cache_value_sorted)  


def intersect_format(A,B):
    ncols = A.shape[1]
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [A.dtype]}
    C = np.intersect1d(A.view(dtype), B.view(dtype))

    mask_A = np.isin(A.view(dtype), C.view(dtype)).all(axis=1)
    return mask_A


def fill_Cache_round_format(Cache_index=None, Cache_value=None, alpha=None, beta=None, result=None, round_int=3):
    if Cache_index is None:
        # Erstellen des Caches mit den ersten Einträgen
        raw_alpha_beta = np.round(np.concatenate((alpha, beta), axis=1), round_int)
        # Cache Runden damit gleiche winkel auftreten können
        # Doppelte herausfiltern
        Cache_index, mask = np.unique(raw_alpha_beta, True , axis=0)
        Cache_value = result[mask,:,:]
        # Sortieren des Caches 
        mask_sort = np.argsort(Cache_index[:,0])
        Cache_index_sorted = Cache_index[mask_sort,:]
        Cache_value_sorted = Cache_value[mask_sort,:,:]

    else:
        # Doppelte herausfiltern
        # Zeit test
        start = time.process_time()
        raw_alpha_beta = np.round(np.concatenate((alpha, beta), axis=1), round_int)
        pot_index, mask = np.unique(raw_alpha_beta, True , axis=0)
        pot_val = result[mask,:,:]
        # überprüfen ob das ergebnis schon im chache ist
        mask_dup = intersect_format(pot_index, Cache_index)
        new_index = pot_index[~mask_dup]
        print(new_index.shape)
        new_val = pot_val[~mask_dup,:,:]
        print("Zeit Teil 1",time.process_time() - start)
        # Sortieren
        start1 = time.process_time()
        mask_sort = np.argsort(new_index[:,0])
        new_index_sorted = new_index[mask_sort,:]
        new_val_sorted = new_val[mask_sort,:,:]
        new_cache_index = np.concatenate((Cache_index, new_index_sorted), axis=0)
        new_cache_value = np.concatenate((Cache_value, new_val_sorted), axis=0)
        mask_sort_final = np.argsort(new_cache_index[:,0])
        Cache_index_sorted = new_cache_index[mask_sort_final,:]
        Cache_value_sorted = new_cache_value[mask_sort_final,:,:]
        print("Zeit Teil 2",time.process_time() - start1)

    return np.array(Cache_index_sorted), np.array(Cache_value_sorted)  


def kegel_indizes_cache(Cache_index=None, Cache_value=None, 
                        range_r=4, x_koord=0, y_koord=0, z_koord=0, alpha=np.array([]), beta=np.array([]), 
                        round_int=2):
    # Liste definieren und Duplikate herausfiltern und Zählen
    raw_alpha_beta = np.round(np.concatenate((alpha, beta), axis=1), round_int)
    pot_index, mask, counts= np.unique(raw_alpha_beta, return_index = True, return_counts = True, axis=0)

    # Liste Sortieren
    mask_sorted = np.argsort(pot_index[:,0])
    pot_index_sorted = pot_index[mask_sorted]
    counts_sorted = counts[mask_sorted]

    # Elemente im Cache auslesen
    if Cache_index is not None:
        m1 = intersect_format(pot_index_sorted, Cache_index)
        m2 = intersect_format(Cache_index, pot_index_sorted)
        in_cache_index = pot_index_sorted[m1]
        not_in_cache = pot_index_sorted[~m1]
        counts_in_cache = counts_sorted[m1] 
        counts_not_in_cache = counts_sorted[~m1] 
        in_cache_val = Cache_value[m2]
        # Restliche elemente normal berechnen
        result = koords_in_kegel_cache(range_r, not_in_cache[:,0],not_in_cache[:,1])
        # Werte außerhalb des Caches hinzufügen
        Cache_index, Cache_value = fill_Cache_round_format(Cache_index, Cache_value, not_in_cache[:,0,None], not_in_cache[:,1,None], result, round_int)
        # zusammnfügen
        res_ges = np.concatenate((in_cache_val, result), axis = 0)
        res_ges[:,0,:] += x_koord
        res_ges[:,1,:] += y_koord
        res_ges[:,2,:] += z_koord
        alpha_beta_ges = np.concatenate((in_cache_index, not_in_cache), axis = 0)
        counts_ges = np.concatenate((counts_in_cache, counts_not_in_cache), axis = 0)
    else:
        res_ges = koords_in_kegel_cache(range_r, pot_index[:,0],pot_index[:,1])
        # Werte außerhalb des Caches hinzufügen
        Cache_index, Cache_value = fill_Cache_round_format(Cache_index, Cache_value, pot_index[:,0,None], pot_index[:,1,None], res_ges, round_int)
        alpha_beta_ges = pot_index
        counts_ges = counts
    
    return (Cache_index, Cache_value, alpha_beta_ges, res_ges, counts_ges)




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


# def get_points(range_r):
#     x = np.arange(-range_r, range_r+1, 1)
#     y = np.copy(x)
#     z = np.copy(x)

#     # Erstelle das Gitter für x, y und z
#     x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
#     mask = (x_grid**2 + y_grid**2 + z_grid**2 <= (range_r+1)**2) & (x_grid**2 + y_grid**2 + z_grid**2 >= (range_r-1)**2)
#     x_mask = x_grid[mask]
#     y_mask = y_grid[mask]
#     z_mask = z_grid[mask]
#     Kugelschale = np.concatenate((x_mask[:,None], y_mask[:,None], z_mask[:,None]), axis=1)
#     beta = np.zeros(Kugelschale.shape[0])*np.nan
#     mask_1 = (Kugelschale[:,1] >= 0)
#     beta[mask_1] = np.arccos(Kugelschale[mask_1,0]/range_r)
#     beta[~mask_1] = 2*math.pi - np.arccos(Kugelschale[~mask_1,0]/range_r)
#     if np.sum(np.isnan(beta)) > 0:
#         print(f"Warning, {np.sum(np.isnan(beta))} nan entries in beta!!!")
#     warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
#     warnings.filterwarnings("ignore", message="invalid value encountered in divide")
#     mask_1 = (Kugelschale[:,0] > 0) & (Kugelschale[:,1] < 0)
#     mask_2 = (Kugelschale[:,0] > 0) & (Kugelschale[:,1] >= 0)
#     mask_3 = (Kugelschale[:,0] < 0)
#     # alpha erstallen als Liste mit nur np.nan als inhalt
#     alpha = np.zeros(Kugelschale.shape[0])*np.nan
#     alpha[mask_1] = np.arctan(-Kugelschale[mask_1,1]/Kugelschale[mask_1,2])
#     alpha[mask_2] = np.arctan(-Kugelschale[mask_2,1]/Kugelschale[mask_2,2]) + 2*math.pi
#     alpha[mask_3] = np.arctan(-Kugelschale[mask_3,1]/Kugelschale[mask_3,2]) + math.pi
#     # Bei beta = pi/2 folgt y = sin(alhpa)*sin(pi/2) = sin(alpha)
#     mask_4 = ((beta >= 1.56e+00) & (beta <= 1.58e+00)) 
#     mask_5 = ((beta >= 4.70e+00) & (beta <= 4.72e+00))
#     mask_final = np.any([mask_4,mask_5], axis=0)
#     alpha[mask_final] = np.arcsin(Kugelschale[mask_final,1]/range_r)
#     mask_nan = np.isnan(alpha)

#     mask_nan = np.isnan(alpha)
#     manuell_lst = np.where(mask_nan)[0]

#     for i in manuell_lst:
#         input_str = math.pi*int(input(f"Bitte geben sie den zu Koordinate: {Kugelschale[i]} und Beta: {beta[i]} passenden Wikel n*\\pi für Alpha an:\n\t"))
#         alpha[i] = input_str

#     return(Kugelschale, alpha, beta)

# def generate_dict(range_r = 10, factor = 10):
#     Kugelschale, alpha, beta = get_points(factor)
#     temp_dict = {tuple(a_row): b_row for a_row, b_row in zip(Kugelschale, koords_in_kegel_cache(range_r, alpha, beta))}
#     return temp_dict


def get_array(dict_cache, key):
    return dict_cache[tuple(key)]


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
    if paralell_bool == False: # langsamer
        arrays = [dict_cache[tuple(key)] for key in koords_rounded_int]
        result = np.array(arrays)
    else:
        result = np.array(Parallel(n_jobs=-1)(delayed(get_array)(dict_cache,key) for key in koords_rounded_int))

    result[:,0,:] += x_koord
    result[:,1,:] += y_koord
    result[:,2,:] += z_koord
    if get_phi_theta == True:
        # Phi und Theta berechnen für ODFs
        phi = np.arccos(z)
        theta = np.arctan2(y,x)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        return (result, phi, costheta, sintheta)

    return result


# def get_point2(range_r):
#     x = np.arange(-range_r, range_r+1, 1)
#     y = np.copy(x)
#     z = np.copy(x)

#     # Erstelle das Gitter für x, y und z
#     x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
#     mask = (x_grid**2 + y_grid**2 + z_grid**2 <= (range_r+1)**2) & (x_grid**2 + y_grid**2 + z_grid**2 >= (range_r-1)**2)
#     x_mask = x_grid[mask]
#     y_mask = y_grid[mask]
#     z_mask = z_grid[mask]
#     Kugelschale = np.concatenate((x_mask[:,None], y_mask[:,None], z_mask[:,None]), axis=1)
#     # beta = np.zeros(Kugelschale.shape[0])*np.nan
#     # mask_1 = (Kugelschale[:,2] >= 0)
#     beta = np.arccos(Kugelschale[:,0]/range_r)
#     # beta[~mask_1] = 2*math.pi - np.arccos(Kugelschale[~mask_1,0]/range_r)
#     if np.sum(np.isnan(beta)) > 0:
#         print(f"Warning, {np.sum(np.isnan(beta))} nan entries in beta!!!")
#     warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
#     warnings.filterwarnings("ignore", message="invalid value encountered in divide")
#     mask_1 = (Kugelschale[:,1]<0) & (Kugelschale[:,2]<0) & (Kugelschale[:,0]!=0)
#     mask_2 = (Kugelschale[:,2]>0) & (Kugelschale[:,0]!=0)
#     mask_3 = (Kugelschale[:,1]>0) & (Kugelschale[:,2]<0)
#     # alpha erstallen als Liste mit nur np.nan als inhalt
#     alpha = np.zeros(Kugelschale.shape[0])
#     alpha[mask_1] = 2*math.pi + np.arcsin(Kugelschale[mask_1,1]/range_r/np.sin(np.arccos(Kugelschale[mask_1,0]/range_r)))
#     alpha[mask_2] = math.pi-np.arcsin(Kugelschale[mask_2,1]/range_r/np.sin(np.arccos(Kugelschale[mask_2,0]/range_r)))
#     alpha[mask_3] = np.arccos(-Kugelschale[mask_3,2]/range_r/np.sin(np.arccos(Kugelschale[mask_3,0]/range_r)))
#     # Bei beta = pi/2 folgt y = sin(alhpa)*sin(pi/2) = sin(alpha)
#     # mask_4 = ((beta >= 1.56e+00) & (beta <= 1.58e+00)) 
#     # mask_5 = ((beta >= 4.70e+00) & (beta <= 4.72e+00))
#     # mask_final = np.any([mask_4,mask_5], axis=0)
#     # alpha[mask_final] = np.arcsin(Kugelschale[mask_final,1]/range_r)
#     mask_nan = np.isnan(alpha)

#     # with open(f'list_nan_values_{range_r}.txt', 'wb') as f:
#     #     f.writelines(Kugelschale[mask_nan])

#     with open(f'list_nan_values_{range_r}.txt', 'w') as f:
#         f.write('\n'.join(str(i) for i in Kugelschale[mask_nan]))

#     manuell_lst = np.where(mask_nan)[0]
#     print(np.sum(mask_nan))
#     return Kugelschale[mask_nan] 

#     for i in manuell_lst:
#         input_str = math.pi*int(input(f"Bitte geben sie den zu Koordinate: {Kugelschale[i]} und Beta: {beta[i]} passenden Wikel n*\\pi für Alpha an:\n\t"))
#         alpha[i] = input_str

#     return(Kugelschale, alpha, beta)


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
    beta = np.arccos(Kugelschale[:,0]/range_r)
    if np.sum(np.isnan(beta)) > 0:
        print(f"Warning, {np.sum(np.isnan(beta))} nan entries in beta!!!")
    # warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
    # warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    # alpha erstallen als Liste mit nur np.nan als inhalt
    alpha = np.arctan2(Kugelschale[:,2],Kugelschale[:,1])+math.pi/2
    # Bei beta = pi/2 folgt y = sin(alhpa)*sin(pi/2) = sin(alpha)
    # mask_4 = ((beta >= 1.56e+00) & (beta <= 1.58e+00)) 
    # mask_5 = ((beta >= 4.70e+00) & (beta <= 4.72e+00))
    # mask_final = np.any([mask_4,mask_5], axis=0)
    # alpha[mask_final] = np.arcsin(Kugelschale[mask_final,1]/range_r)

    mask_nan = np.isnan(alpha)
    manuell_lst = np.where(mask_nan)[0]
    print(np.sum(mask_nan))
    for i in manuell_lst:
        input_str = math.pi*int(input(f"Bitte geben sie den zu Koordinate: {Kugelschale[i]} und Beta: {beta[i]} passenden Wikel n*\\pi für Alpha an:\n\t"))
        alpha[i] = input_str

    return(Kugelschale, alpha, beta)

def generate_dict(range_r = 10, factor = 10):
    Kugelschale, alpha, beta = get_points(factor)
    temp_dict = {tuple(a_row): b_row for a_row, b_row in zip(Kugelschale, koords_in_kegel_cache(range_r, alpha, beta))}
    return temp_dict

# anpassung der schnellen ungenauen berechnung der Kegel, da die Funktion zur bewertung der Punlte kontinuiertlich ist.
def reverse_rotate_and_translate_data(data, x_koord = 0, y_koord = 0, z_koord = 0, alpha = np.array([]), beta = np.array([])):
    # Translation Rückgängig machen
    data[:,0,:] -= x_koord
    data[:,1,:] -= y_koord
    data[:,2,:] -= z_koord
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
    # print(rotation_matrix.shape, data.shape)
    result = np.matmul(rotation_matrix, data)

    return result



