import pandas as pd
import matplotlib.pyplot as plt


import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge, TheilSenRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def modelos_func(modelos, df):
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    dr = pd.DataFrame()

    # Definir X e y
    X = df.drop('dias_diferencia_entrega', axis=1)
    y = df['dias_diferencia_entrega']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49150, test_size=0.2)

    # Escalador
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Loop a través de los modelos
    for i in modelos:
        if i == 'SVR()':
            nombre = 'SVR'
            modelo = SVR()
        elif i == 'RandomForestRegressor()':
            nombre = 'RandomForest'
            modelo = RandomForestRegressor(random_state=49150)
        elif i == 'DecisionTreeRegressor()':
            nombre = 'DecisionTree'
            modelo = DecisionTreeRegressor(random_state=49150)
        elif i == 'XGBRegressor()':
            nombre = 'XGBRegressor'
            modelo = XGBRegressor(random_state=49150)
        elif i == 'LGBMRegressor()':
            nombre = 'LGBMRegressor'
            modelo = LGBMRegressor(random_state=49150,verbose=-1)
        elif i == 'AdaBoostRegressor()':
            nombre = 'AdaBoostRegressor'
            modelo = AdaBoostRegressor(random_state=49150)
        elif i == 'GradientBoostingRegressor()':
            nombre = 'GradientBoostingRegressor'
            modelo = GradientBoostingRegressor(random_state=49150)
        elif i == 'BayesianRidge()':
            nombre = 'BayesianRidge'
            modelo = BayesianRidge()
        elif i == 'ExtraTreesRegressor()':
            nombre = 'ExtraTreesRegressor'
            modelo = ExtraTreesRegressor(random_state=49150)
        elif i == 'HistGradientBoostingRegressor()':
            nombre = 'HistGradientBoostingRegressor'
            modelo = HistGradientBoostingRegressor(random_state=49150)
        elif i == 'TheilSenRegressor()':
            nombre = 'TheilSenRegressor'
            modelo = TheilSenRegressor()
        else:
            raise ValueError(f"Modelo {i} no está soportado.")   # Asegurar que el modelo esté en la lista

        # Ajustar el modelo
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)

        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5  # Calcular la raíz cuadrada de MSE para obtener RMSE
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Crear un DataFrame con los resultados
        nuevoModelo = pd.DataFrame({
            'model': [nombre],
            'mse': [mse],
            'rmse': [rmse],
            'mae': [mae],
            'r2': [r2]
        })

        # Usar pd.concat para añadir el nuevo modelo al DataFrame
        dr = pd.concat([dr, nuevoModelo], ignore_index=True)

    return dr  # Devolver el DataFrame con los resultados



#----------- funcion para tunnear el modelo --------------#

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def entrenar_y_optimizar_modelo(modelo, param_grid, X_train, y_train, X_test, y_test, scoring='r2', cv=5):

    
 
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, 
                               cv=cv, n_jobs=-1, scoring=scoring, verbose=2)
    
   
    grid_search.fit(X_train, y_train)

 
    mejores_parametros = grid_search.best_params_
    print("Mejores hiperparámetros:", mejores_parametros)


    modelo_optimizado = grid_search.best_estimator_

    y_pred_optimizado = modelo_optimizado.predict(X_test)
    

  
    mse_opt = mean_squared_error(y_test, y_pred_optimizado)
    rmse_opt = mse_opt ** 0.5
    mae_opt = mean_absolute_error(y_test, y_pred_optimizado)
    r2_opt = r2_score(y_test, y_pred_optimizado)

   
    print("MSE:", mse_opt)
    print("RMSE:", rmse_opt)
    print("MAE:", mae_opt)
    print("R²:", r2_opt)

    return modelo_optimizado, mejores_parametros

# Ejemplo de uso con cualquier modelo y cualquier conjunto de hiperparámetros:
# from sklearn.ensemble import RandomForestRegressor
# modelo = RandomForestRegressor(random_state=49150)
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }
# modelo_optimizado, mejores_parametros = entrenar_y_optimizar_modelo(modelo, param_grid, X_train_scaled, y_train, X_test_scaled, y_test)

def entrenar_y_optimizar_modelo_grafica(modelo, param_grid, X_train, y_train, X_test, y_test, scoring='r2', cv=5):
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, 
                               cv=cv, n_jobs=-1, scoring=scoring, verbose=2)
    
   
    grid_search.fit(X_train, y_train)

 
    mejores_parametros = grid_search.best_params_
    print("Mejores hiperparámetros:", mejores_parametros)


    modelo_optimizado = grid_search.best_estimator_

    y_pred_optimizado = modelo_optimizado.predict(X_test)
    

  
    mse_opt = mean_squared_error(y_test, y_pred_optimizado)
    rmse_opt = mse_opt ** 0.5
    mae_opt = mean_absolute_error(y_test, y_pred_optimizado)
    r2_opt = r2_score(y_test, y_pred_optimizado)

   
    print("MSE:", mse_opt)
    print("RMSE:", rmse_opt)
    print("MAE:", mae_opt)
    print("R²:", r2_opt)


    plt.figure(figsize=(10,6))
    plt.scatter(y_test,y_pred_optimizado, alpha=0.5, label="Predicciones")
    plt.xlabel("Valores Reales")
    plt.ylabel("Valores Predichos")
    plt.title("Valores Reales vs. Predicciones")
    plt.legend()
    plt.show()


    return modelo_optimizado, mejores_parametros