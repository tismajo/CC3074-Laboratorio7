import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_split(df, target_col, test_size=0.2, random_state=42):
    print("Creando conjuntos de entrenamiento y prueba...")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # importante para clasificación
    )

    # ==============================
    # Guardar índices
    # ==============================
    os.makedirs("./splits", exist_ok=True)

    X_train.index.to_series().to_csv("./splits/train_index.csv", index=False)
    X_test.index.to_series().to_csv("./splits/test_index.csv", index=False)

    print("Split guardado en carpeta ./splits")

    return X_train, X_test, y_train, y_test

def load_split(df, target_col):
    print("Cargando conjuntos previamente guardados...")

    train_idx = pd.read_csv("./splits/train_index.csv").squeeze()
    test_idx = pd.read_csv("./splits/test_index.csv").squeeze()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    print("Split cargado correctamente.")

    return X_train, X_test, y_train, y_test