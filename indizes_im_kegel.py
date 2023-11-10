import numpy as np
import matplotlib.pylab as plt
import math

def indizes_des_Kegels(limit=25, Offset_i=12, Offset_j=12, Offset_k=12, limit_r=7, alpha=0, beta=0):
    # Länge des Kegels
    # limit_r = 10
    # Offset des Kegels in x-Richtung
    # Offset_i, Offset_j, Offset_k = 12,12,12
    # alpha: drehwinkel um die j/x-Achse, beta: drehwinkel um die k/z-Achse
    # Größe der N**3 Matrix
    # limit = 25
    # gedrehtes i, i k:
    # i = (np.cos(beta)*(np.cos(alpha)*i+np.sin(alpha)*k)-np.sin(beta)*j)
    # j = (np.sin(beta)*(np.cos(alpha)*i+np.sin(alpha)*k)+np.cos(beta)*j)
    # k = (np.cos(alpha)*k-np.sin(alpha)*i)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    
    _Index_lst_im_Kegel =[[i+Offset_i,j+Offset_j,k+Offset_k] 
        for i in range(-limit_r*2 if limit_r*2 < Offset_i else -Offset_i, limit_r*2 if limit_r*2 < limit-Offset_i else limit-Offset_i) 
        for j in range(-limit_r*2 if limit_r*2 < Offset_j else -Offset_j, limit_r*2 if limit_r*2 < limit-Offset_j else limit-Offset_j) 
        for k in range(-limit_r*2 if limit_r*2 < Offset_k else -Offset_k, limit_r*2 if limit_r*2 < limit-Offset_k else limit-Offset_k) 
        if (cos_beta*(cos_alpha*i+sin_alpha*k)-sin_beta*j) <= limit_r
        if (sin_beta*(cos_alpha*i+sin_alpha*k)+cos_beta*j)**2 + (cos_alpha*k-sin_alpha*i)**2 <= (cos_beta*(cos_alpha*i+sin_alpha*k)-sin_beta*j)**2
        if (cos_beta*(cos_alpha*i+sin_alpha*k)-sin_beta*j) > 0
        ] 
    
    return _Index_lst_im_Kegel


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
 
    winkel_alpha = float(input("winkel Alpha aneben: \n"))
    winkel_beta = float(input("winkel Beta aneben: \n"))
    indizes_des_Kegels_lst_1 = indizes_des_Kegels(limit=100, Offset_i=50, Offset_j=50, Offset_k=50, limit_r=7, alpha=winkel_alpha * math.pi/180, beta=winkel_beta * math.pi/180)
    xs1, ys1, zs1 = [list(x) for x in zip(*indizes_des_Kegels_lst_1)]
    ax.azim, ax.dist, ax.elev = -60, 10, 30
    ax.scatter(xs1, ys1, zs1)
            
    plt.show()