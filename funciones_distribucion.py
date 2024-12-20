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
from sklearn.utils import resample
#\
import warnings
import inspect

import funciones_eda
from funciones_eda import *





def resumen_distribucion(df,cantidad):
    
    Z = 1.96  # Nivel de confianza del 95%
    p = 0.5   # Proporción estimada
    E = 0.05  # Margen de error
    N = len(df)  # Tamaño de la población
    
    n_infinito = (Z**2 * p * (1 - p)) / E**2  # Muestra para población infinita
    n_finito = (N * n_infinito) / (N + n_infinito - 1)
    
    tamaño_muestra=math.ceil(n_finito)
    

    muestra_aleatoria=df.sample(tamaño_muestra,random_state=42)
    
    media_poblacional =np.mean(df[cantidad])
    media_aleatoria=np.mean(muestra_aleatoria[cantidad])
    
    desviacion_estandar_poblacional=np.std(muestra_aleatoria[cantidad], ddof=1)
    error_estándar = desviacion_estandar_poblacional / np.sqrt(tamaño_muestra)
    
    nivel_confianza = 0.95
    z = stats.norm.ppf(1 - (1 - nivel_confianza) / 2)  # Valor crítico Z
    margen_error = z * error_estándar
    intervalo_confianza = (media_aleatoria - margen_error, media_aleatoria + margen_error)
    
    print(f'poblacion {N}')
    print(f'tamaño de la muestra {tamaño_muestra}')
    print(f'media poblacional {round(media_poblacional,1)}')
    print(f'media aleatoria {round(media_aleatoria,1)}')
    print(f'error estandar {round(error_estándar,1)}')
    print(f'intervalo_confianza {intervalo_confianza}')
    
    if intervalo_confianza[0] <= media_poblacional <= intervalo_confianza[1]:
        print("La media poblacional está dentro del intervalo de confianza.")
    else:
        print("La media poblacional NO está dentro del intervalo de confianza.")
        

def ditribucion_muestral(df,cantidad):
    
    # Parámetros para el cálculo del tamaño de la muestra
    Z = 1.96  # Nivel de confianza del 95%
    p = 0.5   # Proporción estimada
    E = 0.05  # Margen de error
    N = len(df)  # Tamaño de la población

    # Calcular el tamaño de la muestra para población finita
    n_infinito = (Z**2 * p * (1 - p)) / E**2  # Muestra para población infinita
    n_finito = (N * n_infinito) / (N + n_infinito - 1)

    # Redondear el tamaño de la muestra
    tamaño_muestra = math.ceil(n_finito)

    # Seleccionar una muestra aleatoria del DataFrame
    muestra_aleatoria = df.sample(tamaño_muestra, random_state=42)

    # Seleccionar la columna de interés
    poblacion = df[cantidad].values  # Convertir a un arreglo NumPy para trabajar con np.random.choice

    # Número de muestras para la distribución muestral
    num_muestras = len(df)  # Puedes ajustar este valor según lo necesario

    # Generar las medias muestrales
    medias_muestrales = [
        np.mean(np.random.choice(poblacion, tamaño_muestra, replace=True))
        for _ in range(num_muestras)
    ]
    
    sigma = np.std(medias_muestrales, ddof=1)  # Desviación estándar de las medias muestrales
    error_estandar = sigma / np.sqrt(tamaño_muestra)

    return medias_muestrales,error_estandar



def ditribucion_muestral_bossting(df,cantidad):
    
    Z = 1.96  # Nivel de confianza del 95%
    p = 0.5   # Proporción estimada
    E = 0.05  # Margen de error
    N = len(df)  # Tamaño de la población

    # Calcular el tamaño de la muestra para población finita
    n_infinito = (Z**2 * p * (1 - p)) / E**2  # Muestra para población infinita
    n_finito = (N * n_infinito) / (N + n_infinito - 1)

    # Redondear el tamaño de la muestra
    tamaño_muestra = math.ceil(n_finito)
    poblacion = df[cantidad].values
    num_muestras = len(df)
    
    medias_muestrales_con_remplazo = [
        np.mean(np.random.choice(poblacion, tamaño_muestra, replace=True))
        for _ in range(num_muestras)
    ]
    
    medias_muestrales_sin_remplazo = [
        np.mean(np.random.choice(poblacion, tamaño_muestra, replace=False))
        for _ in range(num_muestras)
    ]
    
    return   medias_muestrales_con_remplazo,medias_muestrales_sin_remplazo
    
    
    
def distribucion_cola_larga(df,cantidad):
    
    df_sorted = df.sort_values(by=[cantidad], ascending=False).reset_index(drop=True)

        # Preparar datos para graficar
    products = df_sorted.index  # Índices (productos ordenados)
    sales = df_sorted[cantidad]    # Ventas ordenadas

        # Graficar la distribución de cola larga
    plt.figure(figsize=(10, 6))
    plt.bar(products[:300], sales[:300], color='blue')  # Graficar los primeros 50 productos
    plt.title('Distribución de Cola Larga: Ventas ', fontsize=14)
    plt.xlabel('Productos (ordenados por ventas)', fontsize=12)
    plt.ylabel('Ventas cantidad', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    


def distribuciones_personalizadas(df, variable):
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 8))
    
    # Distribución t de Student
    sns.kdeplot(
        t.pdf(df[variable], df=10),
        fill=True, color="blue", ax=axes[0]
    )
    sns.rugplot(
        t.pdf(df[variable], df=10),
        color="blue", ax=axes[0]
    )
    axes[0].set_title("Distribución t de Student", fontsize='medium')
    axes[0].set_xlabel('t(df=10)', fontsize='small') 
    axes[0].tick_params(labelsize=6)

    # Distribución Binomial
    sns.kdeplot(
        binom.pmf(np.arange(0, len(df[variable])), n=20, p=0.5),
        fill=True, color="blue", ax=axes[1]
    )
    sns.rugplot(
        binom.pmf(np.arange(0, len(df[variable])), n=20, p=0.5),
        color="blue", ax=axes[1]
    )
    axes[1].set_title("Distribución Binomial", fontsize='medium')
    axes[1].set_xlabel('Binomial(n=20, p=0.5)', fontsize='small') 
    axes[1].tick_params(labelsize=6)

    # Distribución Chi-Cuadrado
    sns.kdeplot(
        chi2.pdf(df[variable], df=5),
        fill=True, color="blue", ax=axes[2]
    )
    sns.rugplot(
        chi2.pdf(df[variable], df=5),
        color="blue", ax=axes[2]
    )
    axes[2].set_title("Distribución Chi-Cuadrado", fontsize='medium')
    axes[2].set_xlabel('Chi^2(df=5)', fontsize='small') 
    axes[2].tick_params(labelsize=6)

    # Distribución F
    sns.kdeplot(
        f.pdf(df[variable], dfn=5, dfd=10),
        fill=True, color="blue", ax=axes[3]
    )
    sns.rugplot(
        f.pdf(df[variable], dfn=5, dfd=10),
        color="blue", ax=axes[3]
    )
    axes[3].set_title("Distribución F", fontsize='medium')
    axes[3].set_xlabel('F(dfn=5, dfd=10)', fontsize='small') 
    axes[3].tick_params(labelsize=6)

    fig.tight_layout()
    
    
    
    
def desequilibrio_datos(df,variable):
    
    desq=df[variable].value_counts().reset_index()
    desq.columns = [variable, 'conteo']
    
    plt.figure(figsize=(4, 4))
    plt.bar(desq[variable], desq['conteo'], alpha=0.8)
    plt.xlabel(variable, fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.xticks(rotation=45)
    plt.title(f'Desequilibrio de la variable {variable}', fontsize=14)
    

def equilibrar_datos(df,variable,categoria_1,categoria_2):
    mayoritaria = df[df[variable] == categoria_1]
    minoritaria = df[df[variable] == categoria_2]

    # Submuestreo de la clase mayoritaria
    minoritaria_oversampled = resample(minoritaria, replace=True, n_samples=len(mayoritaria), random_state=42)

    # Concatenar y revisar
    df_balanceado = pd.concat([mayoritaria, minoritaria_oversampled])
    
    return df_balanceado