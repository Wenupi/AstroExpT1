import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

def f_gauss_3D(xy, mean, covariance): #Función para obtener los valores z
    return multivariate_normal.pdf(xy_gaussian, mean, covariance)

# Estos arreglo los saqué de interntet
# https://stackoverflow.com/questions/40622203/how-to-plot-3d-gaussian-distribution-with-matplotlib
x_gaussian, y_gaussian = np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j]
xy_gaussian = np.column_stack([x_gaussian.flat, y_gaussian.flat])

x_g = np.linspace(-1,1,11)
y_g = np.linspace(-1,1,11)
muu = np.array([0, 0])
sigmaa = np.array([1, 1])
cov_g = np.diag(sigmaa**2)
#np.cov(x)

mu_g = np.array([0.0, 0.0])
sigma_g = np.array([.5, .5])
covariance_g = np.diag(sigma_g**2)

"""
Aquí trato de hacer el curve_fit
"""
#params_g, pcov_g = curve_fit(f_gauss_3D, np.column_stack([x_gaussian.flat, y_gaussian.flat]), 
#                            f_gauss_3D(xy_gaussian, mean=mu_g, covariance=covariance_g)) 
                            #los guess que le puse creo que son los valores exactos


"""
Este es un ejemplo que encontré en internet para 
poner un arreglo 2D como el parámetro x
====================================================
"""
A = np.array([(19,20,24), (10,40,28), (10,50,31)])
def func(data, a, b):
    return data[:,0]*data[:,1]*a + b
guess = (1,1)
params, pcov = curve_fit(func, A[:,:2], A[:,2], guess)
"""===================================================
"""

# Con esto se hace el plot sin hacer ningún fit
z_gaussian= multivariate_normal.pdf(xy_gaussian, mean=mu_g, cov=covariance_g)
z_g = multivariate_normal.pdf(np.array([x_g, y_g]), mean=muu, cov=cov_g)

# Reshape back to a (30, 30) grid.
z_gaussian = z_gaussian.reshape(x_gaussian.shape)

# Plot
fig6 = plt.figure()
ax6 = fig6.add_subplot(111, projection='3d')
ax6.plot_surface(x_gaussian,y_gaussian,z_gaussian)
fig6.savefig("Intento_Gaussiana3D")