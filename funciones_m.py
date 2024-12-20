import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge, TheilSenRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt

def modelos_func(modelos, df):
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    dr = pd.DataFrame()

    # Definir X e y
    X = df.drop('dias_diferencia_entrega', axis=1)
    y = df['dias_diferencia_entrega']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49150, test_size=0.2)

    # Escalador
    scaler = StandardScaler()
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

        joblib.dump(modelo, f'C:\\Users\\andre\\OneDrive\\Documentos\\ciencia de datos\\Df_proyecto\\{nombre}_entrenado.pkl')
        joblib.dump(scaler, r'C:\Users\andre\OneDrive\Documentos\ciencia de datos\Df_proyecto\scaler.pkl')
        X_train_columns = X_train.columns
        joblib.dump(X_train_columns, r'C:\Users\andre\OneDrive\Documentos\ciencia de datos\Df_proyecto\X_train_columns.pkl')

        y_pred = modelo.predict(X_test_scaled)
        y_pred_train = modelo.predict(X_train_scaled)


        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5  # Calcular la raíz cuadrada de MSE para obtener RMSE
        mae = mean_absolute_error(y_test, y_pred)

        r2 = r2_score(y_test, y_pred)
        r2_train = r2_score(y_train, y_pred_train)

        std_mse = np.std((y_test - y_pred) ** 2)
        std_mse_ = np.std((y_train - y_pred_train) ** 2)

        # Crear un DataFrame con los resultados
        nuevoModelo = pd.DataFrame({
            'model': [nombre],
            'mse': [mse],
            'rmse': [rmse],
            'mae': [mae],
            'std_mse_test': [std_mse],
            'std_mse_train': [std_mse_],
            'r2_test': [r2],
            'r2_train': [r2_train]
        })

        # Usar pd.concat para añadir el nuevo modelo al DataFrame
        dr = pd.concat([dr, nuevoModelo], ignore_index=True)

    return dr  # Devolver el DataFrame con los resultados


#----------aplicar modelo-----------------#



def modelos_func_hiperparametros(df,modelo,h_parametros):
    # Ignorar warnings futuros
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Definir X e y
    X = df.drop('dias_diferencia_entrega', axis=1)
    y = df['dias_diferencia_entrega']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49150, test_size=0.2)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Definir el modelo con los hiperparámetros específicos
    model = XGBRegressor(**h_parametros,
        random_state=49150
        # bootstrap=False, 
        # criterion='absolute_error', 
        # max_depth=None, 
        # max_features='log2', 
        # min_samples_leaf=1, 
        # min_samples_split=5, 
        # n_estimators=300
    )
    
    # Entrenar el modelo
    model.fit(X_train_scaled, y_train)
    
    # Predecir en los conjuntos de entrenamiento y prueba
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calcular los errores residuales
    residuals = y_test - y_test_pred

    r2 = r2_score(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)

    # Crear el gráfico de Residuales vs Predicciones
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))

    # Gráfico de Residuales vs Predicciones
    # 1. Gráfico de Valores Verdaderos vs. Predichos
    axs[0].scatter(range(len(y_test)), y_test, color='orange', label="Valores de Testeo", s=20)
    axs[0].scatter(range(len(y_test_pred)), y_test_pred, color='blue', label="Valores Predicción", s=20, alpha=0.6)
    axs[0].set_title(f"Valores Verdaderos vs. Predichos\nR2 (Test) = {r2:.3f}")
    axs[0].legend()

    # 2. Gráfico de Residuales vs. Predicciones
    axs[1].scatter(y_test_pred, residuals, alpha=0.6, color='red',label="Valores residuos")
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].set_xlabel("Predicted Values")
    axs[1].set_ylabel("Residuals")
    axs[1].set_title("Residuals vs Predicted Values")
    axs[1].legend()

    # 3. Histograma de Residuales
    axs[2].hist(residuals, bins=30, color='purple', edgecolor='black', alpha=0.7)
    axs[2].set_xlabel("Residuals")
    axs[2].set_ylabel("Frequency")
    axs[2].set_title("Histogram of Residuals")

    plt.tight_layout()
    plt.show()
    
    
    
    #--------- feature importance -------#
    
def importance_1(modelos,df):
    
    df=pd.get_dummies(df,drop_first=1)
        # Definir X e y
    X = df.drop('dias_diferencia_entrega', axis=1)
    y = df['dias_diferencia_entrega']

        # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49150, test_size=0.2)

        # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
        
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
            raise ValueError(f"Modelo {i} no está soportado.") 


        
            # Entrenar el modelo
        modelo.fit(X_train_scaled, y_train)     
        importances = modelo.feature_importances_
        y_pred = modelo.predict(X_test_scaled)


        feature_importances_df = pd.DataFrame({
            'Característica': X_train.columns,
                'Importancia': importances
            })

            # Ordenar el DataFrame por la importancia (de mayor a menor)
        feature_importances_df = feature_importances_df.sort_values(by='Importancia', ascending=False)
        feature_importances_df['acumulado']=round(feature_importances_df['Importancia'].cumsum(),1)
        feature_importances_df['Importancia']=round(feature_importances_df['Importancia'],1)


        feature_importances_df=feature_importances_df.reset_index().head()
        feature_importances_df['Variable_Global'] = feature_importances_df['Característica'].str.split('_').str[0]

        global_importance = feature_importances_df.groupby('Variable_Global')['Importancia'].sum().reset_index()
        global_importance=global_importance.sort_values(by='Importancia',ascending=False)

        #print(f'r2 train: {round(r2_score(y_test, y_pred),1)}')
    return global_importance



#------ resultados----- #

def resultados_(datos):
    resultados=pd.DataFrame()
    conteo=0
    for i in datos:
        conteo+=1
        result=i.iloc[:1]
        result['modelo']=f'modelo_{conteo}'
        resultados = pd.concat([resultados, result], ignore_index=True)
        resultados=resultados.sort_values(by='r2_test', ascending=False)
    return print(resultados)



#----------Residuales vs Predicciones----------#
def Residuales(df):
    # Definir X e y
    X = df.drop('dias_diferencia_entrega', axis=1)
    y = df['dias_diferencia_entrega']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49150, test_size=0.2)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definir el modelo
    model = XGBRegressor(random_state=49150)
    model.fit(X_train_scaled, y_train)

    # Predecir en el conjunto de prueba
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calcular los errores residuales
    residuals = y_test - y_test_pred

    r2 = r2_score(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)

    # Crear el gráfico de Residuales vs Predicciones
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))

    # Gráfico de Residuales vs Predicciones
    # 1. Gráfico de Valores Verdaderos vs. Predichos
    axs[0].scatter(range(len(y_test)), y_test, color='orange', label="Valores de Testeo", s=20)
    axs[0].scatter(range(len(y_test_pred)), y_test_pred, color='blue', label="Valores Predicción", s=20, alpha=0.6)
    axs[0].set_title(f"Valores Verdaderos vs. Predichos\nR2 (Test) = {r2:.3f}")
    axs[0].legend()

    # 2. Gráfico de Residuales vs. Predicciones
    axs[1].scatter(y_test_pred, residuals, alpha=0.6, color='red',label="Valores residuos")
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].set_xlabel("Predicted Values")
    axs[1].set_ylabel("Residuals")
    axs[1].set_title("Residuals vs Predicted Values")
    axs[1].legend()

    # 3. Histograma de Residuales
    axs[2].hist(residuals, bins=30, color='purple', edgecolor='black', alpha=0.7)
    axs[2].set_xlabel("Residuals")
    axs[2].set_ylabel("Frequency")
    axs[2].set_title("Histogram of Residuals")

    plt.tight_layout()
    plt.show()


#-----------Histograma de Residuales-------------#

def modelos_func__(modelos, X_train, X_test, y_train, y_test):
    
    warnings.filterwarnings("ignore", category=FutureWarning)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Loop a través de los modelos
    for i in modelos:
        if i == 'SVC()':
            nombre = 'SVC'
            modelo = SVC(probability=True, random_state=49150)
        elif i == 'RandomForestClassifier()':
            nombre = 'RandomForestClassifier'
            modelo = RandomForestClassifier(class_weight='balanced', random_state=49150)
        elif i == 'DecisionTreeClassifier()':
            nombre = 'DecisionTreeClassifier'
            modelo = DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=49150)
        elif i == 'XGBClassifier()':
            nombre = 'XGBClassifier'
            modelo = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=49150)
        elif i == 'LGBMClassifier()':
            nombre = 'LGBMClassifier'
            modelo = LGBMClassifier(class_weight='balanced', random_state=49150)
        elif i == 'AdaBoostClassifier()':
            nombre = 'AdaBoostClassifier'
            modelo = AdaBoostClassifier(random_state=49150)
        elif i == 'GradientBoostingClassifier()':
            nombre = 'GradientBoostingClassifier'
            modelo = GradientBoostingClassifier(random_state=49150)
        elif i == 'LogisticRegression()':
            nombre = 'LogisticRegression'
            modelo = LogisticRegression(class_weight='balanced', random_state=49150)
        elif i == 'HistGradientBoostingClassifier()':
            nombre = 'HistGradientBoostingClassifier'
            modelo = HistGradientBoostingClassifier(random_state=49150)
        elif i == 'GaussianNB()':
            nombre = 'GaussianNB'
            modelo = GaussianNB()
        else:
            raise ValueError(f"Modelo {i} no está soportado.")   # Asegurar que el modelo esté en la lista

        # Ajustar el modelo
        modelo.fit(X_train_scaled, y_train)

        y_prob_test = modelo.predict_proba(X_test_scaled)
        y_prob_train = modelo.predict_proba(X_train_scaled)

        # Calcular métricas de clasificación (accuracy, precision, recall, F1-score)
        y_pred = modelo.predict(X_test_scaled)
        y_pred_train = modelo.predict(X_train_scaled)


        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Crear un DataFrame con los resultados del modelo
        nuevoModelo = pd.DataFrame({
            'model': [nombre],
            'accuracy': [accuracy],
            'precision': [precision],
            'recall': [recall],
            'f1_score': [f1],
        })

        # Usar pd.concat para añadir el nuevo modelo al DataFrame
        dr = pd.DataFrame(nuevoModelo)

    return dr  # Devolver el DataFrame con los resultados