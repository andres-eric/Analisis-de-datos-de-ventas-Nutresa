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
from sklearn.metrics import silhouette_score

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
    filtrado=df[features].copy()
    clientes_scaled = scaler.fit_transform(filtrado)

    kmeans = KMeans(n_clusters=n, random_state=42)  
    df = df.copy()
    df.loc[:,'Cluster'] = kmeans.fit_predict(clientes_scaled)
    centroids = kmeans.cluster_centers_
    
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
    
    
def centroide(df,n):
    scaler = StandardScaler()
    features = ['SALES_ORDER_ID', 'QTY', 'PRODUCT_ID']
    filtrado=df[features].copy()
    clientes_scaled = scaler.fit_transform(filtrado)

    kmeans = KMeans(n_clusters=n, random_state=42,n_init=10)  
    df = df.copy()
    df.loc[:,'Cluster'] = kmeans.fit_predict(clientes_scaled)
    centroids_scaled = kmeans.cluster_centers_
    
    # Desescalar los centroides al espacio original
    centroids_original = scaler.inverse_transform(centroids_scaled)

    
    return centroids_original

def kmeans_con_silueta(df, feature,min_clusters=2, max_clusters=8):

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[feature])

    silhouette_scores = []
    best_k = min_clusters
    best_score = -1
    best_centroids = None

    for k in range(min_clusters, max_clusters + 1):
        # Entrenar el modelo K-Means
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)

        # Calcular el índice de silueta
        score = silhouette_score(data_scaled, kmeans.labels_)
        silhouette_scores.append(score)

        # Actualizar el mejor modelo
        if score > best_score:
            best_score = score
            best_k = k
            best_centroids = kmeans.cluster_centers_
            best_labels = kmeans.labels_

    # Asignar los clusters al DataFrame original
    df_resultado = df.copy()
    df_resultado['Cluster'] = best_labels

    # Graficar el índice de silueta
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=range(min_clusters, max_clusters + 1), y=silhouette_scores, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Índice de Silueta')
    plt.title('Índice de Silueta para diferentes Clusters')
    plt.show()
    
    
    

def agrupar_closter_producto(df):
    clientes= df.groupby('PRODUCT_ID').agg({
    
   'SALES_ORDER_ID': 'nunique',  # Frecuencia (número de órdenes)
    'QTY': 'sum',                 # Cantidad total comprada
    'CUSTOMER_ID': 'nunique',      # Número de productos únicos
    #'CATEGORY_ID': 'nunique',     # Número de categorías únicas
    #'SUB_CATEGORY_ID': 'nunique', # Número de subcategorías únicas
    #'ORDER_DATE': lambda x: (df['ORDER_DATE'].max())
    #'ORDER_DATE': lambda x: (df['ORDER_DATE'].max() - df['ORDER_DATE'].min()).days    
    })
    return clientes.sort_values(['SALES_ORDER_ID','QTY'],ascending=False)


def metodo_codo_productos(df,n):
    scaler = StandardScaler()
    features = ['SALES_ORDER_ID', 'QTY', 'CUSTOMER_ID']
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
    
    
def modelo_kmeans_productos(df,n):
    scaler = StandardScaler()
    features = ['SALES_ORDER_ID', 'QTY', 'CUSTOMER_ID']
    filtrado=df[features].copy()
    clientes_scaled = scaler.fit_transform(filtrado)

    kmeans = KMeans(n_clusters=n, random_state=42)  
    df = df.copy()
    df.loc[:,'Cluster_producto'] = kmeans.fit_predict(clientes_scaled)
    centroids = kmeans.cluster_centers_
    
    return df