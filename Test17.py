from new_lib import *

# def genereate_divergence_():
#     field_phi = np.empty((50,50,2*range_r+1))
#     field_theta = np.copy(field_phi)
#     for i in range(field_phi.shape[0]):
#         for j in range(field_phi.shape[1]):
#             for k in range(field_phi.shape[2]):
#                 field_phi[i,j,k] = np.arccos((k-25)/np.linalg.norm([i-25,j-25,k-25])) # np.arccos(k+20/np.linalg.norm([i+20,j+20,k+20]))
#                 field_theta[i,j,k] = np.arctan2(j-25,i-25)
#     return(field_theta, field_phi)

def genereate_divergence_(middle_x:int=3,middle_y:int=3,middle_z:int=3):
    field_phi = np.empty((7,7,2*range_r+1))
    field_theta = np.copy(field_phi)
    for i in range(field_phi.shape[0]):
        for j in range(field_phi.shape[1]):
            for k in range(field_phi.shape[2]):
                field_phi[i,j,k] = 0 # np.arccos((k-middle_z)/np.linalg.norm([i-middle_x,j-middle_y,k-middle_z])) # np.arccos(k+20/np.linalg.norm([i+20,j+20,k+20]))
                field_theta[i,j,k] = np.arctan2(j-middle_y,i-middle_x)
    return(field_theta, field_phi)

idex = 250
# Kegel länge
range_r = 10
# Anzahl der Bänder der ODFs und AODFs
bands = 10
# Anzahl der Sampling Punkte
number_of_winkel = 1500
# Gaussfunktion Sigma
sigma = 2
# Faktor, welche Punkte in die AODFs eingehen
factor_amp = 100
# Faktor für die Größe der AODFs in der Visualisierung
coefficient = 0.7
# ODFs Generieren
# Direction_image, Inclination_image, mask_image, rel_image = load_data(300, 310)
# ODFs = odf3.compute(np.deg2rad(Direction_image)[:,:,:,None], np.deg2rad(Inclination_image)[:,:,:,None], mask_image[:,:,:,None], bands)[600:650,900:950,:,:]

field_theta, field_phi = genereate_divergence_()
# field_theta[:,:,:] += np.pi/2
mask = np.ones(field_phi[:,:,:,None].shape)
mask[3,3,:] = 0
ODFs = odf3.compute(field_theta[:,:,:,None], field_phi[:,:,:,None], mask, bands)

limit_x, limit_y, limit_z = ODFs.shape[0],ODFs.shape[1],ODFs.shape[2] #20,20,20# 
print(ODFs.shape)
# Kegel generieren
alpha, beta = fibonacci_sphere_2(number_of_winkel) #get_points_noRand(12, buffer=0.4) 
phi, theta = alpha_to_phi(alpha, beta)
#theta = np.pi/2 - theta
result, basis = get_result_basis(range_r, bands, alpha, beta)
result1, basis = get_result_basis(range_r, bands, alpha, beta)
result_rot = reverse_rotate_and_translate_data_noTranslation(result, alpha, beta)
weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], 0, 0, sigma)


# Kegel nicht rotiert
alpha_extra, beta_extra = np.array([0,math.pi/180*60]),np.array([0,math.pi/180*45])
result_extra, basis_extra = get_result_basis(10, bands, alpha_extra, beta_extra)
result_rot_extra = reverse_rotate_and_translate_data_noTranslation(result_extra, alpha_extra, beta_extra)
weights_extra = gauss_2d(result_rot_extra[:,1,:], result_rot_extra[:,2,:], 0, 0, sigma)

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
print(alpha_extra[0]*180/math.pi,beta_extra[0]*180/math.pi)
ax.scatter(*result_extra[0,:,:])

ax.tick_params(axis='x', labelsize=0)
ax.tick_params(axis='y', labelsize=0)
ax.tick_params(axis='z', labelsize=0)

plt.savefig("AbgerundeterKegel.png", dpi=100)

plt.show()
# Rotierter Kegel
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
print(alpha[idex]*180/math.pi,beta[idex]*180/math.pi)
ax.scatter(*result[idex,:,:])

ax.tick_params(axis='x', labelsize=0)
ax.tick_params(axis='y', labelsize=0)
ax.tick_params(axis='z', labelsize=0)

plt.savefig("AbgerundeterKegel_rotiert.png", dpi=100)

plt.show()


i=idex
color_factor = weights
res = result[i]
mask = ~np.isnan(res)

gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[8, 1])  # Verhältnis der Höhen der Subplots

# Erstelle den 3D Scatterplot als oberen Subplot
ax = plt.subplot(gs[0], projection='3d')


# Farbverlauf ist von Rot(kleines Gewicht) bis Blau(großes Gewicht)
red = Color("red")
colors = list(red.range_to(Color("blue"), int(1000*np.nanmax(color_factor[i])+1)))
for xs,ys,zs,col in zip(*np.reshape(res[mask], (3,-1)),color_factor[i]):
    ax.scatter(xs,ys,zs, c = str(colors[int(1000*col)]))



#Plot Colormap
x_color = range(len(colors))
dict_colors = dict(zip(x_color,colors))
color_f = lambda n : dict_colors.get(n)
# odf3._set_axes_equal(ax)

# Erstelle den Farbverlauf als unteren Subplot
ax1 = plt.subplot(gs[1])
# ax1.imshow(aspect='auto')
for x_arg in x_color:
    ax1.axvline(x_arg, color=str(color_f(x_arg)))

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.get_yaxis().set_ticks([])
ax.tick_params(axis='x', labelsize=0)
ax.tick_params(axis='y', labelsize=0)
ax.tick_params(axis='z', labelsize=0)
ax1.tick_params(axis='x', labelsize=0)

plt.savefig("AbgerundeterKegel_rotiert_gauss_farbig.png", dpi=100)
plt.show()

##################################### AODFs

# Kegel länge
range_r = 3
# Anzahl der Bänder der ODFs und AODFs
bands = 10
# Anzahl der Sampling Punkte
number_of_winkel = 1500
# Gaussfunktion Sigma
sigma = 0.3
# Faktor, welche Punkte in die AODFs eingehen
factor_amp = 100
# Faktor für die Größe der AODFs in der Visualisierung
coefficient = 0.7
# ODFs Generieren
# Direction_image, Inclination_image, mask_image, rel_image = load_data(300, 310)
# ODFs = odf3.compute(np.deg2rad(Direction_image)[:,:,:,None], np.deg2rad(Inclination_image)[:,:,:,None], mask_image[:,:,:,None], bands)[600:650,900:950,:,:]

field_theta, field_phi = genereate_divergence_()
# field_theta[:,:,:] += np.pi/2
mask = np.ones(field_phi[:,:,:,None].shape)
mask[3,3,:] = 0
ODFs = odf3.compute(field_theta[:,:,:,None], field_phi[:,:,:,None], mask, bands)

limit_x, limit_y, limit_z = ODFs.shape[0],ODFs.shape[1],ODFs.shape[2] #20,20,20# 
print(ODFs.shape)
# Kegel generieren
alpha, beta = fibonacci_sphere_2(number_of_winkel) #get_points_noRand(12, buffer=0.4) 
phi, theta = alpha_to_phi(alpha, beta)
#theta = np.pi/2 - theta
result, basis = get_result_basis(range_r, bands, alpha, beta)
result_rot = reverse_rotate_and_translate_data_noTranslation(result, alpha, beta)
weights = gauss_2d(result_rot[:,1,:], result_rot[:,2,:], 0, 0, sigma)



# AODFs Generieren
AODFs = np.array([
    Get_AODF_noRand_noCache_amp(ODFs,result, basis, weights,phi,theta,i,j,k, sigma=sigma, factor_amp=factor_amp, bands=bands)[0]
    for i in tqdm(range(range_r, limit_x - range_r))
    for j in range(range_r, limit_y - range_r)
    for k in range(range_r, limit_z - range_r)
])

Amps = np.array([
    Get_AODF_noRand_noCache_amp(ODFs,result, basis, weights,phi,theta,i,j,k, sigma=sigma, factor_amp=factor_amp, bands=bands)[1]
    for i in tqdm(range(range_r, limit_x - range_r))
    for j in range(range_r, limit_y - range_r)
    for k in range(range_r, limit_z - range_r)
])
# np.save(f"Amps_Stern_NoRand_nocache_b{bands}_s{sigma*10}_fib_2", Amps)
length = 1
AODFs = np.reshape(AODFs,(length,length,length,odf.get_num_coeff(bands)))

odf.visualize_odf(AODFs[0,0,0,:],120,120)
plt.show()



odf_coeff = AODFs
print(odf_coeff.shape)
odf_coeff = odf_coeff[:, :, 0, :].squeeze()
print(odf_coeff.shape)

# image = vispy_odf.render_scene(odf_coeff*coefficient)
# plt.imshow(image)

# # plt.savefig(f"new_lib_fib2_b{bands}_s{int(sigma*10)}_c{coefficient}_famp{factor_amp}_n{number_of_winkel}_r{range_r}_fib_2.png", dpi=500)
# plt.show()

Amps = np.reshape(Amps, (1,1,1,1500))


i = idex
print(Amps[0,0,0,idex])
x = Amps[0,0,0,:] * np.cos(phi) * np.sin(theta)
y = Amps[0,0,0,:] * np.sin(phi) * np.sin(theta)
z = Amps[0,0,0,:] * np.cos(theta)

x_f = Amps[0,0,0,:] * np.cos(phi) * np.sin(theta) * factor_amp
y_f = Amps[0,0,0,:] * np.sin(phi) * np.sin(theta) * factor_amp
z_f = Amps[0,0,0,:] * np.cos(theta) * factor_amp

origin = [0,0,0]
X, Y, Z = zip(origin,origin,origin) 

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.quiver(X,Y,Z,x_f[idex],y_f[idex],z_f[idex], color="tab:blue")
ax.scatter(x[idex], y[idex], z[idex])
ax.scatter(*result1[idex,:,:], alpha=0.3, color="tab:orange")
# 1010,1005
idex = 1030
ax.quiver(X,Y,Z,x_f[idex],y_f[idex],z_f[idex], color="tab:blue")
ax.scatter(x[idex], y[idex], z[idex])
ax.scatter(*result1[idex,:,:], alpha=0.3, color="tab:orange")

ax.tick_params(axis='x', labelsize=0)
ax.tick_params(axis='y', labelsize=0)
ax.tick_params(axis='z', labelsize=0)

plt.savefig("Amplitude_vec.png", dpi=100)

for i in range(1):
    x = Amps[0,i,0,:] * np.cos(phi) * np.sin(theta)
    y = Amps[0,i,0,:] * np.sin(phi) * np.sin(theta)
    z = Amps[0,i,0,:] * np.cos(theta)


    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    ax.scatter(x, y, z)
ax.tick_params(axis='x', labelsize=0)
ax.tick_params(axis='y', labelsize=0)
ax.tick_params(axis='z', labelsize=0)

plt.savefig("Amplitude_ges.png", dpi=100)
plt.show()


odf_coeff = AODFs
print(odf_coeff.shape)
odf_coeff = odf_coeff[:, :, 0, :].squeeze()
print(odf_coeff.shape)

image = vispy_odf.render_scene(odf_coeff*coefficient)
plt.imshow(image)
plt.show()