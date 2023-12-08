from main_lib import *



def Get_AODF(ODFs:np.ndarray, dict_res:dict, dict_basis:dict, factor:int, x_koord:int, y_koord:int, z_koord:int, bands:int=10, number_of_winkel:int = 1000, factor_amp:int = 1000):
    # Winkel generieren
    rng = np.random.default_rng(random.randint(100000,10000000000))
    beta = np.arccos(1-2*(rng.random(number_of_winkel).reshape(number_of_winkel,1)))
    alpha = rng.random(number_of_winkel).reshape(number_of_winkel,1)*math.pi*2

    # Punkte im Kegel mit Basisvektoren und Winkeln generieren
    result, basis, phi, theta = kegel_from_dict_withBasis(dict_res, dict_basis, factor, x_koord, y_koord, z_koord, alpha, beta, True)
    result_rot = reverse_rotate_and_translate_data(result, x_koord, y_koord, z_koord, alpha, beta)
    weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], y_koord, z_koord, sigma = 2)

    # Mit weights alle punkte im Kegel ablaufen
    AODF_Amplitude = get_amplitude(result, ODFs, basis, weights)

    ## Scatterplot generieren
    # x = AODF_Amplitude[:,None] * np.cos(phi) * np.sin(theta)
    # y = AODF_Amplitude[:,None] * np.sin(phi) * np.sin(theta)
    # z = AODF_Amplitude[:,None] * np.cos(theta)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection = "3d")
    # n = 10000
    # ax.scatter(x[:n],y[:n],z[:n])
    # plt.show()


    num_greater = np.sum(np.rint(AODF_Amplitude[AODF_Amplitude > 0]*100))
    
    num_greater = int(np.sum(np.rint(AODF_Amplitude[AODF_Amplitude > 0]*factor_amp)))
    multiple_dir = np.empty((num_greater,1))*np.nan
    multiple_inc = np.empty((num_greater,1))*np.nan

    count = 0
    for i,j in enumerate(AODF_Amplitude):
        for k in range(int(j*factor_amp if j > 0 else 0)):
            multiple_dir[count,:] = phi[int(i)]
            multiple_inc[count,:] = np.pi/2 - theta[int(i)] 
            count += 1

    nan_mask = ~np.isnan(multiple_inc)
    AODF_d = odf.compute(np.ravel(multiple_dir[nan_mask])[None,None,None,:],np.ravel(multiple_inc[nan_mask])[None,None,None,:], np.ones(np.ravel(multiple_inc[nan_mask])[None,None,None,:].shape), bands)
    return AODF_d