import pandas as pd
import pyreadr
import os

def run_eda(archive):
    path = "./data/" + archive

    print("Procesando archivo...")

    # detectar tipo de archivo
    if archive.endswith(".RData"):
        result = pyreadr.read_r(path)
        df = result["listings"]

    elif archive.endswith(".csv"):
        df = pd.read_csv(path)

    else:
        raise ValueError("Formato no soportado. Usa .RData o .csv")

    # === EDA ===
    print(f"Tamaño del dataframe: \n{df.shape}\n")
    print(f"Primeras 5 filas: \n{df.head()}\n")
    print(f"Resumen: \n{df.describe()}\n")

    print("Columnas:")
    df.info()

    nulos = df.isnull().sum()
    porcentaje = (nulos / len(df)) * 100

    resumen_nulos = pd.DataFrame({
        'nulos': nulos,
        'porcentaje': porcentaje
    }).sort_values(by='nulos', ascending=False)

    print(f"\nNulos y porcentajes: \n{resumen_nulos}\n")
    print(f"\nCantidad de duplicados: {df.duplicated().sum()}\n")

    return df