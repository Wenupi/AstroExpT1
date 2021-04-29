import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

df_HCT = pd.read_excel('/Users/javier/Documents/University/5th semester/Experimental astronomy/'
                     'Tarea_1_Instrumentaci_n_Radioastron_mica/HCT_AE2020A.xlsx', comment='#')
df_antdip = pd.read_excel('/Users/javier/Documents/University/5th semester/'
                     'Experimental astronomy/Tarea_1_Instrumentaci_n_Radioastron_mica/antdip_AE2021A.xlsx',
                      comment='#')              


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

#print(T_rec_exp)
path = '/Users/javier/Documents/University/5th semester/Experimental astronomy/Tarea_1_Instrumentaci_n_Radioastron_mica/sec_mierc_sem1_2021/'
x=1
v,T = np.genfromtxt(path+'sdf_11'+str(x)+'_11'+str(x), unpack = True, skip_header=108)

def f_gauss(x,T0,mean,stdv):
    return T0*np.exp(-((x-mean)**2)/(2*(stdv**2)))

fg = [20, 10, 1] 
plt.figure(1)

coefs,cov = curve_fit(f_gauss,v,T, p0=fg) # Se fitea

t0, M, S = coefs[0],coefs[1],coefs[2]  # Se extraen los coeficientes fiteados

#print ('Valores fiteados: (t0,M,S) =',(t0,M,S))

#plt.plot(v,T)
#plt.title('Espectro', fontsize=18)
#plt.xlabel('Velocidad [$\frac{km}{s}$]', fontsize=18)
#plt.show()
#plt.ylabel('Temperatura [K]', fontsize=18)

plt.plot(v,T, label='Data')
plt.plot(v,f_gauss(v,t0, M,S), label='Fiteo')
plt.title('Espectro', fontsize=18)
plt.ylabel('Temperatura [K]', fontsize=18)
plt.xlabel(r'Velocidad [$\frac{km}{s}$]', fontsize=18)
plt.legend()
plt.show()