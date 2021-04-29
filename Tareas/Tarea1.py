import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import glob

df_HCT = pd.read_excel('/Users/javier/Documents/University/5th semester/Experimental astronomy/'
                     'Tarea_1_Instrumentaci_n_Radioastron_mica/HCT_AE2020A.xlsx', comment='#')
df_antdip = pd.read_excel('/Users/javier/Documents/University/5th semester/'
                     'Experimental astronomy/Tarea_1_Instrumentaci_n_Radioastron_mica/antdip_AE2021A.xlsx',
                      comment='#')              

"""
Pregunta 1
"""
def dBm_to_Watts(P):
    output = 10**((P-30)/10)
    return output


def T_rec(T_H, T_C, y):
    output = (T_H-y*T_C)/(y-1)
    return output

T_hot = df_HCT.iloc[2, 4]
T_cold = df_HCT.iloc[3, 4]
Y = dBm_to_Watts(float(df_HCT.iloc[2, 1]))/dBm_to_Watts(float(df_HCT.iloc[3, 1]))

T_rec_exp = T_rec(T_hot, T_cold, Y)
print(T_rec_exp)

"""
Pregunta 2
"""
def lineal(x, m, a):
    x=np.array(x)
    y= m*x +a
    return y

DeltaOmega_ln = df_antdip.iloc[:-1,7]
secante = df_antdip.iloc[:-1,3]

z = np.polyfit(secante, DeltaOmega_ln, 1)
x_lineal = [np.min(secante), np.max(secante)]
y_lineal = lineal(x_lineal, z[0], z[1])

plt.figure(1)
plt.rcParams["font.family"] = "serif"
plt.clf()
plt.plot(x_lineal, y_lineal, 'k')
plt.scatter(secante, DeltaOmega_ln)
plt.title(r'Fiteo $\tau_w$')
plt.xlabel('-sec(z)')
plt.ylabel(r'ln($\Delta$ W)')
plt.grid()
plt.show()

"""
Pregunta 3
"""
path = '/Users/javier/Documents/University/5th semester/Experimental astronomy/Tarea_1_Instrumentaci_n_Radioastron_mica/sec_mierc_sem1_2021/'
x=1
v,T = np.genfromtxt(path+'sdf_11'+str(x)+'_11'+str(x), unpack = True, skip_header=108)

def f_gauss(x,T0,mean,stdv):
    return T0*np.exp(-((x-mean)**2)/(2*(stdv**2)))

fg = [20, 10, 1] 
plt.figure(2)

coefs,cov = curve_fit(f_gauss,v,T, p0=fg) # Se fitea

t0, M, S = coefs[0],coefs[1],coefs[2]  # Se extraen los coeficientes fiteados

#print ('Valores fiteados: (t0,M,S) =',(t0,M,S))

#plt.plot(v,T)
#plt.title('Espectro', fontsize=18)
#plt.xlabel('Velocidad [$\frac{km}{s}$]', fontsize=18)
#plt.show()
#plt.ylabel('Temperatura [K]', fontsize=18)

fl = sorted(glob.glob(path+'sdf*'))

#print(fl)
#for i in range(len(fl)):
#    v,T = np.genfromtxt(fl[i], unpack = True, skip_header=108)
#    plt.plot(v,T)
#    plt.title('Espectro '+fl[i][-11:], fontsize=18)
#    plt.ylabel('Temperatura [K]', fontsize=18)
#    plt.xlabel(r'Velocidad [$\frac{km}{s}$]', fontsize=18)
#    plt.show()

plt.plot(v,T, label='Data')
plt.plot(v,f_gauss(v,t0, M,S), label='Fiteo')
plt.title('Espectro', fontsize=18)
plt.ylabel('Temperatura [K]', fontsize=18)
plt.xlabel(r'Velocidad [$\frac{km}{s}$]', fontsize=18)
plt.legend()
plt.show()