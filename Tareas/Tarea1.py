import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import glob
from scipy.integrate import trapz
from matplotlib.font_manager import FontProperties

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
print('El valor de $T_{REC}$ es:', T_rec_exp)


"""
Pregunta 2
"""
def lineal(x, m, a):
    x=np.array(x)
    y= m*x +a
    return y

DeltaOmega_ln = df_antdip.iloc[:-1,7]
secante_negativa = df_antdip.iloc[:-1,3]

z = np.polyfit(secante_negativa, DeltaOmega_ln, 1)
x_lineal = [np.min(secante_negativa), np.max(secante_negativa)]
y_lineal = lineal(x_lineal, z[0], z[1])
tau_0 = z[0]

font = FontProperties()
font.set_family('serif')

fig1, ax1 = plt.subplots(figsize=(5, 3.5))
ax1.plot(x_lineal, y_lineal, color='#4a9923', label='Ajuste lineal')
ax1.scatter(secante_negativa, DeltaOmega_ln, marker='x', color='black', label='Datos')
ax1.set_title(r'Fiteo $\tau_w$', fontsize=15, fontproperties=font)
ax1.set_xlabel('-sec($E_l$)', fontsize=12, fontproperties=font)
ax1.set_ylabel(r'ln($\Delta$ W)', fontsize=12, fontproperties=font)

ax1.text(-6, -13.75, '$y=$'+str('%.3f'%(tau_0))+'x'+str('%.3f'%(z[1])), fontsize=10)
ax1.legend()
ax1.grid()
fig1.tight_layout()    
fig1.savefig("Fiteo_tau_w")


"""
Pregunta 3
"""
path_observation = '/Users/javier/Documents/University/5th semester/'\
                   'Experimental astronomy/Tarea_1_Instrumentaci_n_Radioastron_mica/sec_mierc_sem1_2021/'


def f_gauss(x,T0,mean,stdv):
    """Función que define la campana de Gauss
    """
    return T0*np.exp(-((x-mean)**2)/(2*(stdv**2)))

# Creación de una figura para colocar los 15 plots en una grilla de 5x3

fig2, ax2 = plt.subplots(5, 3, figsize=(15,20))

# Se define la lista vacía T_max para guardar el nombre del sdf con su
# temperatura máxima y coordenadas galácticas

T_maxs=[]

# Para colocar los gráficos en la figura 2 se definen las position_i que indica
# la columna en la que se colocará para su fila correspondiente
position1 = 0
position2 = 0
position3 = 0
position4 = 0
position5 = 0

# Iteración para realizar el mismo procedimiento para los archivos localizados en
# path_observation
for i in range(11, 26):
    # Se abre el archivo y se leen todas las líneas que contiene

    file = open(path_observation+'sdf_1'+str(i)+'_1'+str(i), "r")
    lines = file.readlines()

    # Seleccionamos las filas donde se encuentran las coordenadas galácticas

    lii = float(lines[22].strip()[5:])
    bii = float(lines[23].strip()[5:])

    # Se seleccionan las columnas que contienen las velocidades y temperaturas

    v, T = np.genfromtxt(path_observation+'sdf_1'+str(i)+'_1'+str(i), 
                         unpack=True, skip_header=108)

    # Para realizar el fiteo se necesita una aproximación de los
    # parámetros: temperatura máxima, media y desviación estándar

    fg = [np.max(T), 10, 1]  #first guess

    # Con curve fit se obtienen los valores reales para realizar el fiteo

    coefs, cov = curve_fit(f_gauss,v,T, p0=fg)
    t0, M, S = coefs[0],coefs[1],coefs[2]  # valor de temperatura máxima, media y desviación estándar

    # Se añaden los valores mencionados anteriormente, en T_max

    T_maxs.append(('sdf_1'+str(i), t0, lii, bii))

    # Debido a que se realizaron mediciones en 5 posiciones diferentes, se ocupan condicionales
    # para ver a qué posición corresponde cada medición.
    # Los plots del espectro real y el fiteo gaussianos se van agregan al objeto ax2 según las 
    # coordenadas galácticas y el número de la medición (primera medición -> 0, segunda " -> 1...)

    if T_maxs[i-11][2:4] == (208.996002, -19.260527):  # primera posición
        ax2[0, position1].plot(v, T, color='black')
        ax2[0, position1].plot(v, f_gauss(v, t0, M, S), color='#4a9923')
        ax2[0, position1].set_ylim((-1,32))
        position1 += 1
    elif T_maxs[i-11][2:4] == (208.863495, -19.385527):  # segunda posición
        ax2[1, position2].plot(v, T, color='black')
        ax2[1, position2].plot(v, f_gauss(v, t0, M, S), color='#4a9923')
        ax2[1, position2].set_ylim((-1,32))
        position2 += 1
    elif T_maxs[i-11][2:4] == (208.996002, -19.385527):  # tercera posición
        ax2[2, position3].plot(v, T, color='black')
        ax2[2, position3].plot(v, f_gauss(v, t0, M, S), color='#4a9923')
        ax2[2, position3].set_ylim((-1,32))
        position3 += 1
    elif T_maxs[i-11][2:4] == (209.12851, -19.385527):  # cuarta posición
        ax2[3, position4].plot(v, T, color='black')
        ax2[3, position4].plot(v, f_gauss(v, t0, M, S), color='#4a9923')
        ax2[3, position4].set_ylim((-1,32))
        position4 += 1
    elif T_maxs[i-11][2:4] == (208.996002, -19.510527):  # quinta posición
        ax2[4, position5].plot(v, T, color='black')
        ax2[4, position5].plot(v, f_gauss(v, t0, M, S), color='#4a9923')
        ax2[4, position5].set_ylim((-1,32))
        position5 += 1
    else:  # por si algo sale mal
        None

# Se guarda el plot final con todos los subplots
ax2[0,1].set_title('Espectro de Orión', fontsize=30, fontproperties=font)
ax2[2,0].set_ylabel('Temperatura', fontsize=25, fontproperties=font)
ax2[4,1].set_xlabel('Velocidad', fontsize=25, fontproperties=font)
fig2.tight_layout()    
fig2.savefig("Espectros")