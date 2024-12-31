import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

# EDA functions
def get_null_percentage_per_column(df):
    null_percentages = (df.isnull().sum() / len(df)) * 100
    return null_percentages.sort_values(ascending=False)


def turn_column_into_date(df, column):
    df[column] = pd.to_datetime(df[column], format='%d/%m/%Y')
    df.set_index(column, inplace=True)
    return df


def interpolate_df(df):
    df.interpolate(method='time', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    print(df.isnull().sum())
    return df


def normalize_dataframe(df):
    scaler = MinMaxScaler()
    df[df.select_dtypes(include=['number']).columns] = scaler.fit_transform(df.select_dtypes(include=['number']))
    return df


def remove_columns_with_nulls_percent(df, threshold):
    null_percentage = (df.isnull().sum() / len(df)) * 100
    columns_to_drop = null_percentage[null_percentage > threshold].index
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned


# Graphic functions
def plot_trend(df, period):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    plt.figure(figsize=(12, 8))
    
    for column in numeric_columns:
        try:
            decomposition = seasonal_decompose(df[column], model="additive", period=period)
            plt.plot(df.index, decomposition.trend, label=f"Trend - {column}")
        except ValueError as e:
            print(f"Error processing column {column}: {e}")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title("Tendencia de los datos")
    plt.xlabel("Fecha")
    plt.ylabel("Tendencia")
    plt.grid()
    plt.show()


def plot_seasonality(df, period):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    plt.figure(figsize=(12, 8))
    
    for column in numeric_columns:
        try:
            decomposition = seasonal_decompose(df[column], model="additive", period=period)
            plt.plot(df.index, decomposition.seasonal, label=f"Seasonality - {column}")
        except ValueError as e:
            print(f"Error processing column {column}: {e}")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title("Estacionalidad de los datos")
    plt.xlabel("Fecha")
    plt.ylabel("Estacionalidad")
    plt.grid()
    plt.show()


def test_stationarity(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    for column in numeric_columns:
        timeseries = df[column]
        print(f"Resultados de la prueba de Dickey-Fuller para la columna '{column}':")
        dftest = adfuller(timeseries, autolag="AIC")
        dfoutput = pd.Series(dftest[0:4], index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
        
        for key, value in dftest[4].items():
            dfoutput[f"Critical Value ({key})"] = value
        
        print(dfoutput)
        print("-" * 80)


def plot_variability(df, period):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    for column in numeric_columns:
        ts = df[column]
        decomposition = seasonal_decompose(ts, model='additive', period=period)
        
        residual = decomposition.resid
        
        fig, axis = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=ts, label="Serie Temporal Original", ax=axis)
        sns.lineplot(data=residual, label="Residuos", ax=axis)
        
        plt.title(f"Análisis de la Variabilidad: Serie Temporal vs Residuos para la columna '{column}'")
        plt.tight_layout()
        plt.legend()
        plt.show()


def plot_autocorrelation(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    for column in numeric_columns:
        ts = df[column]
        plot_acf(ts)
        
        plt.title(f"Análisis de la Autocorrelación para la columna '{column}'")
        plt.tight_layout()
        plt.show()