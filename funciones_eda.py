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
#\
import warnings
import inspect

import funciones_eda
from funciones_eda import *
    
    
#-------- datos generales--------------#    
def datos_generales(df):
    numero_de_datos=df.shape[0]
    numero_de_features=df.shape[1]
    print(f'El dataframe tiene {numero_de_datos} filas y {numero_de_features} columnas')


def borrar_caracteristicas(df, columnas):
    df=df.copy()
    for col in columnas:
        df.drop(col, axis=1, inplace=True)
    return df 

     
    
def categoricas(df):
    df_cat = df.select_dtypes(include = \
        ["object", 'category']).columns.tolist()

    print(f'Las columnas categoricas son: {len(df_cat)}')

    return df_cat
    
    
def numericas(df):
    df_num=df.select_dtypes(include = \
        ['float64','float64','int64','int32','int16']).columns.tolist()
    print(f'Las columnas numericas son: {len(df_num)}')
    return df_num

def valores_unicos_columna(df,columna):
    unica=df[columna].unique()
    return unica
    

def resumen_inicial(df):
    tabla_resumen=pd.DataFrame(

    {
        'columns':df.columns,
        'tipo de dato': [ df[i].dtype for i in df.columns   ],
        'categorias': [ df[i].nunique()   for i in df.columns ]
    })
    return tabla_resumen

def valores_unicos_categorias(df):
    df_cat = df.select_dtypes(include = \
        ["object", 'category']).columns.tolist()
    for i in df_cat:
        print(f'{i}: {df[i].nunique()}')
    
def nullos(df):
    nullo = df.isnull().sum().reset_index()
    nullo.columns = ['variable', 'conteo']
    nullo=nullo[nullo.conteo!=0].iloc[:,0]
    columna=[ i  for i in nullo ]
    filtro=df[df[columna].isnull().any(axis=1)]
    print(f'la cantidad de nulos es: {sum(df.isnull().sum())}')
    return filtro
    
def duplicados(df):
    print(f'la cantidad de duplicados es: {df.duplicated().sum()}')
    
def transfor_fecha(df, columnas):
    for col in columnas:
        df[col] = pd.to_datetime(df[col])


def columnas(df):
    columnas = []
    for col in df.columns:
        columnas.append(col)
        
    return columnas

def vacios(x):
    if x == '  ':
        return np.nan  # Retorna None (nulo)
    else:
        return x
    
def imputar(df):
    # Crear un imputador para la media
    columnas=[ i for i in df.columns]
    tipos_originales=df.dtypes

    imputer_mean = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer_mean.fit_transform(df), columns=columnas)

    for i in columnas:
        if tipos_originales[i]=='float64' or tipos_originales[i] == 'int64':
            df[i]=pd.to_numeric(df[i], errors='coerce')

    return df

def tipo_funciones():
    funciones = [func[0] for func in inspect.getmembers(funciones_eda, inspect.isfunction)]
    funciones=[ i for i in funciones]
    return funciones

def tabla__(df,variable):
    tabla=df[variable].value_counts().reset_index()
    tabla['acumulado']=round((tabla['count'].cumsum())/tabla['count'].sum()*100,1)
    #tabla=tabla[tabla['acumulado']<=80]
    return tabla


def shapiro_test(variable):
    data = variable
    stat, p_value = shapiro(data)
    print(f"Estadístico: {stat}")
    print(f"P-valor: {p_value}")
    alpha = 0.05
    if p_value > alpha:
        print("Los datos tienen una distribución normal (no se rechaza H0)")
    else:
        print("Los datos no tienen una distribución normal (se rechaza H0)")

def kolmogorov_smirnov_test(data):
    # Realizar la prueba KS para verificar normalidad
    stat, p_value = kstest(data, 'norm')
    
    # Imprimir los resultados
    print(f"Estadístico KS: {stat}")
    print(f"P-valor: {p_value}")
    
    # Interpretar el resultado
    alpha = 0.05
    
   
    if p_value > alpha:
        print("No se rechaza la hipótesis nula: los datos podrían seguir una distribución normal")
    else:
        print("Se rechaza la hipótesis nula: los datos no siguen una distribución normal")


def agrupar_target(df, agrupar, target):
    df = df.groupby(agrupar)[target].mean().to_dict()
    return df


def clasificacion (x):
    if x >= 0.1:
        return 0
    elif x >= 0.02:
        return 1
    else:
        return 2

def centrales(df,variable):
    resultados = pd.DataFrame(columns=['Variable', 'Media', 'Mediana', 'Moda', 'Desviación Estándar'])
    for col in variable:
        mean = df[col].mean()
        median = df[col].median()
        mode = df[col].mode()[0]
        std = df[col].std()
    
        temp_df = pd.DataFrame({
            'Variable': [col],
            'Media': [mean],
            'Mediana': [median],
            'Moda': [mode],
            'Desviación Estándar': [std]
            
        })
    
    # Concatenar al DataFrame resultados
        if not temp_df.isna().all(axis=None):
            resultados = pd.concat([resultados, temp_df], ignore_index=True)
    
    return resultados


def moda_(df,variables):
    
    resultados = pd.DataFrame(columns=['Variable', 'Moda'])
    for i in variables:
        temp_df = pd.DataFrame({
            'Variable': i,
            'Moda': [df[i].mode()[0]],
            
            
        })
        if not temp_df.isna().all(axis=None):
            resultados = pd.concat([resultados, temp_df], ignore_index=True)
    
    return resultados

def media_robusta(df, variable):
   
    return np.median(df[variable])


    
#-------------------------------------------------------------------------------------#
#El target encoding debe realizarse después de dividir el dataset en entrenamiento 
#y prueba para evitar la fuga de datos (data leakage).
def entrenamiento_target(df, variable,test_size=0.2, random_state=49150):
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    X = df.drop(variable, axis=1)
    y = df[variable]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49150, test_size=0.2)
    
    train_df = X_train.copy()
    train_df['QTY'] = y_train
    #--------------------------------- CUSTOMER -------------------------
    customer_target_encoding = train_df.groupby('CUSTOMER_ID')['QTY'].mean().to_dict()
    
    X_train['CUSTOMER_ID_encoded'] = X_train['CUSTOMER_ID'].map(customer_target_encoding)
    X_test['CUSTOMER_ID_encoded'] = X_test['CUSTOMER_ID'].map(customer_target_encoding)
    #---------------------------------- PRODUCT -------------------------
    
    product_target_encoding = train_df.groupby('PRODUCT_ID')['QTY'].mean().to_dict()
    # Aplicar el encoding en el conjunto de entrenamiento y prueba
    X_train['PRODUCT_ID_encoded'] = X_train['PRODUCT_ID'].map(product_target_encoding)
    X_test['PRODUCT_ID_encoded'] = X_test['PRODUCT_ID'].map(product_target_encoding)
    
    #---------------------------------------------  # -------------------
    global_mean_qty = y_train.mean()
    X_test['CUSTOMER_ID_encoded'].fillna(global_mean_qty, inplace=True)
    X_test['PRODUCT_ID_encoded'].fillna(global_mean_qty, inplace=True)
    
    # Eliminar las columnas originales 'CUSTOMER_ID' y 'PRODUCT_ID' de X_train y X_test
    X_train = X_train.drop(['CUSTOMER_ID', 'PRODUCT_ID'], axis=1)
    X_test = X_test.drop(['CUSTOMER_ID', 'PRODUCT_ID'], axis=1)
    
    # Alinear los conjuntos de entrenamiento y prueba para que tengan las mismas columnas
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    return X_train, X_test, y_train, y_test



        
#-------- outliers  --------------# 

def eliminar_outliers_zscore(df, columna, umbral=2):
    """
    Elimina outliers utilizando el método Z-Score en una columna específica de un DataFrame.
    
    :param df: DataFrame sobre el cual se aplicará el filtro
    :param columna: Nombre de la columna en la cual se calculará el Z-Score
    :param umbral: Umbral para definir outliers (por defecto es 2)
    :return: DataFrame filtrado sin outliers
    """
    # Calcular la media y la desviación estándar
    media = df[columna].mean()
    desviacion_std = df[columna].std()

    # Calcular el Z-Score
    z_score = ((df[columna] - media) / desviacion_std).abs()

    # Identificar outliers
    outliers = df[z_score > umbral]
    
    # Filtrar el DataFrame eliminando los outliers
    df_filtrado = df[z_score <= umbral]
    
    return df_filtrado,outliers


def eliminar_outliers_iqr(df, columna, multiplicador=1.5):
    

    """
    Elimina outliers utilizando el método del Rango Intercuartílico (IQR) en una columna específica.
    
    :param df: DataFrame sobre el cual se aplicará el filtro
    :param columna: Nombre de la columna en la cual se calculará el IQR
    :param multiplicador: Multiplicador para definir los límites (por defecto es 1.5, pero se puede ajustar)
    :return: DataFrame filtrado sin outliers y DataFrame con outliers
    """
    # Calcular el primer y tercer cuartil
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)

    # Calcular el rango intercuartílico (IQR)
    IQR = Q3 - Q1

    # Definir los límites para identificar outliers
    limite_inferior = Q1 - multiplicador * IQR
    limite_superior = Q3 + multiplicador * IQR

    # Identificar los outliers
    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]

    # Filtrar el DataFrame eliminando los outliers
    df_filtrado = df[(df[columna] >= limite_inferior) & (df[columna] <= limite_superior)]
    
    return df_filtrado,outliers

    

#-------- analisis exploratrio-----------------#   

def graficar_barras_columnas(df, columnas, tipo_agrupacion='count', valor=None, paleta='viridis'):
    """
    Genera gráficos de barras para las columnas especificadas en el DataFrame.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        columnas (list): Lista de nombres de columnas a graficar.
        tipo_agrupacion (str): Tipo de agrupación ('count' o 'sum').
        valor (str): Nombre de la columna a sumar (solo si tipo_agrupacion es 'sum').
    """
    num_columnas = len(columnas)
    num_filas = math.ceil(num_columnas / 3)
    
    sns.set_style("whitegrid")
 
    
    fig, ax = plt.subplots(num_filas, 3, figsize=(15, 5 * num_filas))
    ax = ax.flatten()
    for i, column in enumerate(columnas):
        datos = df[column].value_counts().reset_index().head(10)
        datos.columns = [column, 'Frecuencia']
        datos=datos.sort_values(by='Frecuencia')
        sns.barplot(data=datos, x=column, y='Frecuencia', ax=ax[i], palette=paleta,hue=column, dodge=False, legend=False)
        ax[i].set_ylabel("Frecuencia")
        ax[i].set_xlabel(column)
        ax[i].set_title(f'Gráfico de Barras de {column}')
        ax[i].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.show()


    
#-------- analisis exploratrio-----------------#   

def barras_80(df,variable, paleta='viridis'):
   warnings.filterwarnings("ignore", message="Passing `palette` without assigning `hue` is deprecated")
   datos = df[variable].value_counts().reset_index().head(40)
   datos.columns = [variable, 'Frecuencia']
   datos=datos.sort_values(by='Frecuencia',ascending=False)
   datos['porcentaje']=round((datos['Frecuencia']/datos['Frecuencia'].sum())*100,2)
   datos['Frecuencia_acumulada'] = round(datos['porcentaje'].cumsum(),2)
   
   categoria_80 = datos[datos['Frecuencia_acumulada'] <= 80][variable].iloc[-1]

   plt.figure(figsize=(10, 6))
   sns.barplot(data=datos, x=variable, y='Frecuencia', palette="viridis", dodge=False, legend=False, hue=variable,order=datos[variable])
   plt.ylabel("Frecuencia")
   plt.xlabel(variable)
   plt.title(f'Gráfico de Barras de {variable}')
   plt.xticks(rotation=90)  # Rotar etiquetas si son largas

   plt.axvline(x=list(datos[variable]).index(categoria_80), color='red', linestyle='--', label='80% acumulado')   

   plt.tight_layout()
   plt.show()


def tabla_(df,variable, paleta='viridis',had=5):
    datos = df[variable].value_counts().reset_index().head(had)
    datos.columns = [variable, 'Frecuencia']
    datos=datos.sort_values(by='Frecuencia',ascending=False)
    datos['porcentaje']=round((datos['Frecuencia']/datos['Frecuencia'].sum())*100,2)
    datos['Frecuencia_acumulada'] = round(datos['porcentaje'].cumsum(),2)
    
    plt.figure(figsize=(7, 8))
    sns.barplot(data=datos, x=variable, y='Frecuencia', palette="viridis", dodge=False, legend=False, hue=variable,order=datos[variable])
    plt.ylabel("Frecuencia")
    plt.xlabel(variable)
    plt.title(f'Gráfico de Barras de {variable}')
    plt.xticks(rotation=90)  # Rotar etiquetas si son largas

     

    plt.tight_layout()
    plt.show()
    
    return datos[[variable,'Frecuencia_acumulada']]



#--------- distribucion de variables-----------------#

def plot_normal_distribution(df, variables):
    """
    Genera gráficos de distribución para múltiples características y los compara con una distribución normal teórica.

    Parámetros:
    - df: DataFrame que contiene los datos.
    - variables: Lista de nombres de columnas (str) en df que se quieren analizar.
    """
    # Configuración de subgráficos
    fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=(8, 4 * len(variables)))

    for i, variable in enumerate(variables):
        ax = axes[i] if len(variables) > 1 else axes  # Para manejar el caso de un solo gráfico

        # Histograma con KDE
        sns.histplot(df[variable], kde=True, color="skyblue", bins=30, stat="density", label="Datos", ax=ax)

        # Ajuste a una distribución normal
        media = df[variable].mean()
        desviacion_estandar = df[variable].std()
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, media, desviacion_estandar)

        # Curva de la distribución normal teórica
        ax.plot(x, p, 'r--', linewidth=2, label=f'Distribución  (μ={media:.2f}, σ={desviacion_estandar:.2f})')

        # Título y etiquetas
        ax.set_title(f'{variable} vs. Distribución Normal')
        ax.set_xlabel(variable)
        ax.set_ylabel('Densidad')
        ax.legend()

    plt.tight_layout()
    plt.show()


#-------------box_plot-----------------#
def plot_boxplot_normal_comparison(df, variables):
    """
    Genera gráficos de boxplot para múltiples características y los compara con estadísticas de una distribución normal teórica.

    Parámetros:
    - df: DataFrame que contiene los datos.
    - variables: Lista de nombres de columnas (str) en df que se quieren analizar.
    """
    # Configuración de subgráficos
    fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=(8, 4 * len(variables)))

    for i, variable in enumerate(variables):
        ax = axes[i] if len(variables) > 1 else axes  # Para manejar el caso de un solo gráfico

        # Boxplot
        sns.boxplot(x=df[variable], color="lightgreen", ax=ax)

        # Líneas de la media y desviación estándar de la distribución normal teórica
        media = df[variable].mean()
        desviacion_estandar = df[variable].std()
        ax.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media (μ={media:.2f})')
        ax.axvline(media - desviacion_estandar, color='blue', linestyle='--', linewidth=1, label=f'σ={desviacion_estandar:.2f}')
        ax.axvline(media + desviacion_estandar, color='blue', linestyle='--', linewidth=1)

        # Título y etiquetas
        ax.set_title(f'{variable} - Boxplot y Media Normal Teórica')
        ax.set_xlabel(variable)
        ax.set_ylabel('Valor')
        ax.legend()

    plt.tight_layout()
    plt.show()



#----------------------graficos de lineas meses--------------------#
def lineas_meses_año(df, variable):
    
    
    df = df.copy()
    
    df['año']=df[variable].dt.year
    df = df[df['año'] > 2019]
    anio=[i for i in df['año'].unique()]
    
    meses = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
            7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }
    
    plt.figure(figsize=(10, 6))
    
    
    for i in anio:
        data=df[df['año']==i]
        data=data[variable].dt.month.value_counts().reset_index()
        data.sort_values(by=variable,ascending=True)
        
        data['Mes'] = data[variable].map(meses)
        data=data.sort_values(by=variable,ascending=True)
        data=data[['Mes','count']]
        
        sns.lineplot(data=data, x='Mes', y='count', marker='o',label=str(i))
    
    
    sns.lineplot(data=data, x='Mes', y='count', marker='o')
    plt.ylabel("Frecuencia")
    plt.xlabel("Mes")
    plt.title("Frecuencia de ocurrencias por mes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    


#----------------------graficos de barras por años--------------------#    
def grafica_de_barras_compracion(df,paleta='viridis'):
    
    #------crea data set de comparación------#
    
    anio_incumplido = df[df['cumplio'] != 1]
    anio_incumplido = anio_incumplido['año_aproximado'].value_counts().reset_index()
    anio_incumplido.columns = ['año', 'frecuencia_incumplido']
    anio_incumplido = anio_incumplido.sort_values(by='año')
    anio_incumplido = anio_incumplido[(anio_incumplido['año'] >= 2019) & (anio_incumplido['año'] <= 2024)]
    
    anio_cumplido = df[df['cumplio'] != 0]
    anio_cumplido = anio_cumplido['año_aproximado'].value_counts().reset_index()
    anio_cumplido.columns = ['año', 'frecuencia_cumplido']
    anio_cumplido = anio_cumplido.sort_values(by='año')
    anio_cumplido = anio_cumplido[(anio_cumplido['año'] >= 2019) & (anio_cumplido['año'] <= 2024)]
    
    df_comparacion = pd.merge(anio_incumplido, anio_cumplido, on='año', how='outer').fillna(0)
    
    #------crea data set de comparación------#
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35  # Ancho de las barras
    
    x = df_comparacion['año']
    
    ax.bar(x - width/2, df_comparacion['frecuencia_incumplido'], width, label='Incumplido', color=plt.cm.get_cmap(paleta)(0.6))
    ax.bar(x + width/2, df_comparacion['frecuencia_cumplido'], width, label='Cumplido', color=plt.cm.get_cmap(paleta)(0.3))
    
    # Etiquetas y título
    ax.set_xlabel('Año')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Comparación de Cumplimiento e Incumplimiento por Año (>=2019)')
    ax.legend()

    # Mostrar el gráfico
    plt.show()
    
    
#-------------- histograma------------------#

def histograma(df,bi=30):
    plt.hist(df, bins=bi, edgecolor='black')
    plt.axvline(np.mean(df), color='red', linestyle='dashed', linewidth=1.5, label='Media de las Medias')# 'bins' es el número de barras en el histograma
    plt.xlabel('Valores')
    plt.ylabel('Frecuencia')
    plt.title(f'Histograma ')
    plt.grid(True)
    # Mostrar el histograma
    plt.show()


#------------- boxplot--------------#

def boxplot(df,variable):
    plt.figure(figsize=(6, 6))

    # Crear el boxplot
    sns.boxplot(y=df[df[variable]!=0][variable])

    # Añadir título y etiqueta
    plt.title("Distribución de cantidad")

    # Mostrar el gráfico
    plt.show()


#----------- QQ PLOT-------------------#

def qq_plot(variable):
    plt.figure(figsize=(8, 6))
    stats.probplot(variable, dist="norm", plot=plt)
    plt.title("QQ Plot para el Análisis de Normalidad")
    plt.xlabel("Cuantiles Teóricos")
    plt.ylabel("Cuantiles de los Datos")
    plt.grid(True)
    plt.show()
    
    
#-------- scater plot-------#

def scater_plot(df,x,y):
    sns.scatterplot(data=df, x=x, y=y)
    plt.xlabel(f'{x}')
    plt.ylabel(f'{y}')
    plt.show()
    
 
 #--------------- diostribuciones cuadratica log------------------
    
def distribuciones_cuadratica_log(df,variable):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))
    sns.kdeplot(
        df[variable],
        fill    = True,
        color   = "blue",
        ax      = axes[0]
    )
    sns.rugplot(
        df[variable],
        color   = "blue",
        ax      = axes[0]
    )
    axes[0].set_title("Distribución original", fontsize = 'medium')
    axes[0].set_xlabel('precio', fontsize='small') 
    axes[0].tick_params(labelsize = 6)

    sns.kdeplot(
        np.sqrt(df[variable]),
        fill    = True,
        color   = "blue",
        ax      = axes[1]
    )
    sns.rugplot(
        np.sqrt(df[variable]),
        color   = "blue",
        ax      = axes[1]
    )
    axes[1].set_title("Transformación raíz cuadrada", fontsize = 'medium')
    axes[1].set_xlabel('sqrt(precio)', fontsize='small') 
    axes[1].tick_params(labelsize = 6)

    sns.kdeplot(
        np.log(df[variable]),
        fill    = True,
        color   = "blue",
        ax      = axes[2]
    )
    sns.rugplot(
        np.log(df[variable]),
        color   = "blue",
        ax      = axes[2]
    )
    axes[2].set_title("Transformación logarítmica", fontsize = 'medium')
    axes[2].set_xlabel('log(precio)', fontsize='small') 
    axes[2].tick_params(labelsize = 6)

    fig.tight_layout()
    
    
    
    
def distribucion_empirica(df, variables):
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 5))
    axes = axes.flat
    
    columnas=[variables]
    for i,colum in enumerate(columnas):
        sns.histplot(
        data     = df,
        x        = colum,
        stat     = "count",
        kde      = True,
        color    = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        line_kws = {'linewidth': 2},
        alpha    = 0.3,
        ax       = axes[i]
        )
        axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
        axes[i].tick_params(labelsize = 6)
        axes[i].set_xlabel("")

    fig.tight_layout()
    plt.subplots_adjust(top = 0.9)
    fig.suptitle('Distribución variables numéricas', fontsize = 10, fontweight = "bold");
    
    
def tabla_porcentaje(df, variables,cantidad):
        agrupado=df.groupby([variables])[cantidad].sum().reset_index().sort_values(by=[cantidad], ascending=False)
        total=df[cantidad].sum()
        agrupado['porcentaje']=(agrupado[cantidad]/total)*100
        agrupado['Frecuencia_acumulada'] = round(agrupado['porcentaje'].cumsum(),2)
        return agrupado