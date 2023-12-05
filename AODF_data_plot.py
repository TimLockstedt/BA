from main_lib import *

def test(result:np.ndarray, ODFs:np.ndarray, basis:np.ndarray, weights:np.ndarray)->np.ndarray:
    AODF_Amplitude = np.empty((result.shape[0]))
    for i, res in enumerate(result):
        # mask für nan werte
        nan_mask = ~np.isnan(res)
        mask = np.array(res[nan_mask], int)
        # shape anpassen
        mask_3d = np.reshape(mask, (3, int(mask.shape[0]/3)))
        # nur weights auswählen die nicht nan inhalte haben
        weight = weights[i, nan_mask[0]]
        Test_ODF_masked = np.multiply(weight, ODFs[mask_3d[0],mask_3d[1],mask_3d[2]-5].T).T
        current = np.multiply(basis[i], Test_ODF_masked)
        sum = np.sum(current)
        # if sum != 0.0:
        # print(sum,result.shape[0],res[0,~np.isnan(res[0])].shape[0])
        AODF_Amplitude[i] = sum/res[0,~np.isnan(res[0])].shape[0]
    return AODF_Amplitude


# direction füllen
Direction_zero = np.zeros((100,100,10))
Inclination_zero = np.copy(Direction_zero)
mask = np.ones((100,100,10))
for i in range(Direction_zero.shape[0]):
    for j in range(Direction_zero.shape[1]):
        for k in range(Direction_zero.shape[2]):
            if (i-50)**2+(j-50)**2  <= 45**2 and (i-50)**2+(j-50)**2 >= 40**2 and i <=50 and j <=50:
                Direction_zero[i,j,k] = np.arctan2(-i,j)+np.pi/2
            else:
                mask[i,j,k] = 0


band = 12
ODFs = odf3.compute(Direction_zero[:,:,:,None], Inclination_zero[:,:,:,None], mask[:,:,:,None], band)

range_r, factor = 4, 100
dict_10_50 = generate_dict(range_r, factor) 



x_koord, y_koord, z_koord = 16,25,5
number_of_winkel = 10000

rng = np.random.default_rng(random.randint(100000,10000000000))
beta = np.arccos(1-2*(rng.random(number_of_winkel).reshape(number_of_winkel,1)))
alpha = rng.random(number_of_winkel).reshape(number_of_winkel,1)*math.pi*2

result, phi, theta = kegel_from_dict(dict_10_50, factor, x_koord, y_koord, z_koord, alpha, beta, get_phi_theta=True)
costheta, sintheta = np.cos(theta), np.sin(theta)
result_rot = reverse_rotate_and_translate_data(result, x_koord, y_koord, z_koord, alpha, beta)
weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], y_koord, z_koord, sigma = 2)


basis = get_basis(phi, costheta, sintheta, band)


AODF_Amplitude = test(result, ODFs, basis, weights)

x = AODF_Amplitude[:,None] * np.cos(phi) * sintheta
y = AODF_Amplitude[:,None] * np.sin(phi) * sintheta
z = AODF_Amplitude[:,None] * costheta
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
n = 1000
ax.scatter(x[:n],y[:n],z[:n])


import odf

AODF_Dir = phi
AODF_Incl = theta


AODF = odf.compute(np.ravel(AODF_Dir)[None,None,None,:],np.ravel(AODF_Incl)[None,None,None,:], np.ones(np.ravel(AODF_Dir)[None,None,None,:].shape))

fig, ax = odf.visualize_odf(AODF[0,0,0], 64, 64)
plt.show()