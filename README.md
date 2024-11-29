## **Proyecto de Ciencia de Datos: Modelo de Recomendación de Productos Semanal - Grupo Nutresa**

Descripción del proyecto
Este proyecto tiene como objetivo desarrollar un modelo de recomendación de productos semanales basado en datos históricos de ventas del Grupo Nutresa. El modelo ayudará a prever qué productos tienen mayor probabilidad de ser comprados por cada cliente en una semana específica, optimizando la preparación de los equipos comerciales y mejorando la experiencia del cliente.

La base de datos incluye información detallada sobre transacciones históricas, clientes y productos. El proyecto se estructura en cuatro cuadernos principales, diseñados para organizar el flujo de trabajo de manera modular y clara.

Estructura del repositorio
El siguiente repositorio se encuentra dividido en las secciones:

**1. Análisis exploratorio de datos (EDA)**
Archivo: propductos_eda.ipynb

Este cuaderno contiene un análisis exploratorio de datos para entender la estructura y calidad de la base de datos. Las principales tareas realizadas en este cuaderno incluyen:

Revisión de los tipos de datos y valores nulos.
Análisis descriptivo inicial de las principales variables.
Visualización de patrones generales de comportamiento, como tendencias de ventas por cliente y producto.


**2. Distribución de Datos y Muestreo**
Archivo: Distribuciones de datos y muestreo.ipynb

En este cuaderno se analiza la distribución de las variables principales y se realiza el muestreo de los datos. Las tareas principales incluyen:

Análisis de distribuciones de variables relevantes como ventas ( QTY), clientes y productos.
Identificación de sesgos en los datos.
Creación de muestras balanceadas y representativas para pruebas y modelos.


**3. Pruebas significativas**
Archivo: pruebas significativas.ipynb

Aquí se aplican pruebas estadísticas para validar diferencias clave entre grupos y categorías de datos. Las tareas incluyen:

Pruebas de hipótesis para evaluar diferencias significativas entre segmentos de clientes o productos (por ejemplo, prueba Mann-Whitney U).
Verificación de la importancia de variables categóricas y numéricas en el comportamiento de compra.
Interpretación de resultados estadísticos y su implicación para el modelo.

**4. Modelos de recomendación**
Archivo: ?

Este cuaderno contiene la implementación del modelo de recomendación de productos.
