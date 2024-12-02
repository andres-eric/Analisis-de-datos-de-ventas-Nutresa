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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#\
import warnings
import inspect

import funciones_eda
from funciones_eda import *



def agrupar_closter(df):
    clientes= df.groupby('CUSTOMER_ID').agg({
    
   'SALES_ORDER_ID': 'nunique',  # Frecuencia (número de órdenes)
    'QTY': 'sum',                 # Cantidad total comprada
    'PRODUCT_ID': 'nunique',      # Número de productos únicos
    #'CATEGORY_ID': 'nunique',     # Número de categorías únicas
    #'SUB_CATEGORY_ID': 'nunique', # Número de subcategorías únicas
    #'ORDER_DATE': lambda x: (df['ORDER_DATE'].max())
    #'ORDER_DATE': lambda x: (df['ORDER_DATE'].max() - df['ORDER_DATE'].min()).days    
    })
    return clientes.sort_values(['SALES_ORDER_ID','QTY'],ascending=False)


def modelo_kmeans(df,n):
    scaler = StandardScaler()
    features = ['SALES_ORDER_ID', 'QTY', 'PRODUCT_ID']
    clientes_scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=n, random_state=42)
    df['Cluster'] = kmeans.fit_predict(clientes_scaled)
    return df

def metodo_codo(df,n):
    scaler = StandardScaler()
    features = ['SALES_ORDER_ID', 'QTY', 'PRODUCT_ID']
    clientes_scaled = scaler.fit_transform(df[features])
    #centroids = k_means.cluster_centers_
    
    inertias = []
    for k in range(1, n):  # Probar de 1 a 9 clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(clientes_scaled)
        inertias.append(kmeans.inertia_)

    # Graficar el método del codo
    plt.plot(range(1, n), inertias, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    plt.show()
    
