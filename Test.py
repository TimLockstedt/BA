costheta = None
sintheta = None
X = None
phi = None


elif order == -2 :
	return X * np.sin(2 * phi) * sintheta**2 * (221 * costheta**7 - 195 * costheta**4 + 39 * costheta**2 - 1)
		
		
elif order == 2 :
	return -X * np.cos(2 * phi) * sintheta**2 * (221 * costheta**7 - 195 * costheta**4 + 39 * costheta**2 - 1)