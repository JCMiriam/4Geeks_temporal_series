
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

# Función para obtener el porcentaje de valores nulos por columna
def get_null_percentage_per_column(df):
    null_percentages = (df.isnull().sum() / len(df)) * 100
    return null_percentages.sort_values(ascending=False)

# Convertir columna de fecha y establecer como índice
def turn_column_into_date(df, column):
    df[column] = pd.to_datetime(df[column], format='%d/%m/%Y', errors='coerce')
    df.set_index(column, inplace=True)
    return df

# Manejo avanzado de valores faltantes
def interpolate_df(df):
    df.interpolate(method='linear', limit_direction='both', inplace=True)  # Interpolación lineal en ambas direcciones
    df.fillna(method='ffill', inplace=True)  # Relleno hacia adelante
    df.fillna(method='bfill', inplace=True)  # Relleno hacia atrás
    
    missing_summary = df.isnull().sum()
    print("Valores nulos después de la interpolación:")
    print(missing_summary[missing_summary > 0])
    
    return df

# Normalizar datos numéricos
def normalize_dataframe(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# Eliminar columnas con más del umbral de valores nulos
def remove_columns_with_nulls_percent(df, threshold=50):
    null_percentage = (df.isnull().sum() / len(df)) * 100
    columns_to_drop = null_percentage[null_percentage > threshold].index.tolist()
    df_cleaned = df.drop(columns=columns_to_drop)
    print(f"Columnas eliminadas por alto porcentaje de valores nulos ({threshold}%): {columns_to_drop}")
    return df_cleaned

# Combinación de columnas usando la media
def combine_columns_mean(df, columns, new_column):
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df[new_column] = df[columns].mean(axis=1)
    return df

# Visualización de tendencias
def plot_trend(df, period=365):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    plt.figure(figsize=(12, 8))
    
    for column in numeric_columns:
        try:
            decomposition = seasonal_decompose(df[column].dropna(), model="additive", period=period)
            plt.plot(df.index, decomposition.trend, label=f"Tendencia - {column}")
        except Exception as e:
            print(f"Error en tendencia para {column}: {e}")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title("Tendencia de los datos")
    plt.xlabel("Fecha")
    plt.ylabel("Tendencia")
    plt.grid()
    plt.show()

# Visualización de estacionalidad
def plot_seasonality(df, period=365):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    plt.figure(figsize=(12, 8))
    
    for column in numeric_columns:
        try:
            decomposition = seasonal_decompose(df[column].dropna(), model="additive", period=period)
            plt.plot(df.index, decomposition.seasonal, label=f"Estacionalidad - {column}")
        except Exception as e:
            print(f"Error en estacionalidad para {column}: {e}")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title("Estacionalidad de los datos")
    plt.xlabel("Fecha")
    plt.ylabel("Estacionalidad")
    plt.grid()
    plt.show()

# Prueba de Dickey-Fuller para estacionariedad
def test_stationarity(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    for column in numeric_columns:
        timeseries = df[column].dropna()
        
        if len(timeseries) < 30:  # Evitar errores con series demasiado cortas
            print(f"Omitiendo {column}, muy pocos datos para la prueba de Dickey-Fuller.")
            continue
        
        print(f"Resultados de la prueba de Dickey-Fuller para '{column}':")
        dftest = adfuller(timeseries, autolag="AIC")
        dfoutput = pd.Series(dftest[0:4], index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
        
        for key, value in dftest[4].items():
            dfoutput[f"Critical Value ({key})"] = value
        
        print(dfoutput)
        print("-" * 80)

# Análisis de la variabilidad
def plot_variability(df, period=365):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    for column in numeric_columns:
        ts = df[column].dropna()
        
        if len(ts) < period:
            print(f"Omitiendo {column}, muy pocos datos para el análisis de variabilidad.")
            continue
        
        decomposition = seasonal_decompose(ts, model='additive', period=period)
        
        residual = decomposition.resid.dropna()
        
        fig, axis = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=ts, label="Serie Temporal Original", ax=axis)
        sns.lineplot(data=residual, label="Residuos", ax=axis)
        
        plt.title(f"Análisis de la Variabilidad: Serie vs Residuos ({column})")
        plt.tight_layout()
        plt.legend()
        plt.show()

# Análisis de autocorrelación
def plot_autocorrelation(df, lags=40):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    for column in numeric_columns:
        ts = df[column].dropna()
        
        if len(ts) < lags:
            print(f"Omitiendo {column}, muy pocos datos para la autocorrelación.")
            continue
        
        plt.figure(figsize=(10, 5))
        plot_acf(ts, lags=lags)
        
        plt.title(f"Análisis de la Autocorrelación ({column})")
        plt.tight_layout()
        plt.show()
