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