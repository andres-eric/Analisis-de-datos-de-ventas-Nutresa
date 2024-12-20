import pandas as pd
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.impute import SimpleImputer
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import kstest
from sklearn.model_selection import train_test_split
from scipy.stats import t, binom, chi2, f
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import random
#\
import warnings
import inspect

import funciones_eda
from funciones_eda import *



def Prueba_A_B(df,columna,variable,filtro_1,filtro_2):
    grupo_A = df[df[columna] == filtro_1][variable]
    grupo_B = df[df[columna] == filtro_2][variable]

    # 3. Calcular tasas de conversión
    conversion_A = grupo_A.mean()
    conversion_B = grupo_B.mean()

    print(f"Tasa de conversión Grupo A {filtro_1}: {conversion_A:.2%} ")
    print(f"Tasa de conversión Grupo B {filtro_2}: {conversion_B:.2%}")

    # 4. Prueba estadística (t-test)
    t_stat, p_value = ttest_ind(grupo_A, grupo_B)

    print("\nPrueba T:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.22f}")

    if p_value < 0.05:
        print("Conclusión: Hay una diferencia estadísticamente significativa entre los grupos.")
    else:
        print("Conclusión: No hay suficiente evidencia para afirmar que los grupos son diferentes.")
        
        
        
def Prueba_mannwhitneyu(df,columna,variable,filtro_1,filtro_2):
    grupo_A = df[df[columna] == filtro_1][variable]
    grupo_B = df[df[columna] == filtro_2][variable]

    # 3. Calcular tasas de venta
    conversion_A = grupo_A.mean()
    conversion_B = grupo_B.mean()

    print(f"Tasa de venta Grupo A {filtro_1}: {conversion_A:.2%} ")
    print(f"Tasa de venta Grupo B {filtro_2}: {conversion_B:.2%}")

    # 4. Prueba estadística (t-test)
    stat, p = mannwhitneyu(grupo_A, grupo_B, alternative='two-sided')

    print("Estadístico U:", stat)
    print("Valor p:", p)

    if p > 0.05:
        print("No se rechaza la hipótesis nula: las distribuciones son iguales.")
    else:
        print("Se rechaza la hipótesis nula: las distribuciones son diferentes.")
        

def Prueba_mannwhitneyu_bidireccional(df,columna,variable,filtro_1,filtro_2):
    grupo_A = df[df[columna] == filtro_1][variable]
    grupo_B = df[df[columna] == filtro_2][variable]

    # 3. Calcular tasas de venta
    conversion_A = grupo_A.mean()
    conversion_B = grupo_B.mean()

    print(f"Tasa de venta Grupo A {filtro_1}: {conversion_A:.2%} ")
    print(f"Tasa de venta Grupo B {filtro_2}: {conversion_B:.2%}")

    stat, p = mannwhitneyu(grupo_A, grupo_B, alternative='less')

    # Nivel de significancia
    alpha = 0.05

    # Resultado con conclusión
    if p < alpha:
        conclusion = "Se rechaza la hipótesis nula: el grupo 1 tiene valores significativamente menores que el grupo 2."
    else:
        conclusion = "No se rechaza la hipótesis nula: no hay evidencia suficiente para concluir que el grupo 1 es menor que el grupo 2."

    print(f"Estadístico U: {stat}")
    print(f"Valor p: {p}")
    print(f"Conclusión: {conclusion}")

        

def permutacion(a,b):
    
    x = pd.Series(a + b)  # `x` contiene todos los datos combinados

    # Tamaños de los grupos
    nA = len(a)  # 6
    nB = len(b)

    n = nA + nB  # Total de datos combinados
    idx_B = set(random.sample(range(n), nB))  # Índices seleccionados para el grupo B
    idx_A = set(range(n)) - idx_B  # El resto de índices van al grupo A
    return x.loc[list(idx_B)].mean() - x.loc[list(idx_A)].mean()



def grafica_permutaciones(a, b):
    # Calcular diferencia observada
    
    promedio_a = np.mean(a)
    promedio_b = np.mean(b)
    diferencia_observada = promedio_a - promedio_b

    # Generar permutaciones
    n_permutaciones = 500
    diferencias_permutadas = []

    for _ in range(n_permutaciones):
        diferencia = permutacion(a, b)  # Llamar a la función de permutación
        diferencias_permutadas.append(diferencia)

    # Graficar resultados
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(diferencias_permutadas, bins=11, rwidth=0.9, color='blue', alpha=0.7, label="Permutaciones")
    ax.axvline(x=diferencia_observada, color='black', linestyle='--', linewidth=2, label="Diferencia observada")
    ax.set_title("Distribución de Diferencias (Permutaciones)")
    ax.set_xlabel("Diferencia de Medias")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    plt.show()
    
    return diferencias_permutadas,diferencia_observada