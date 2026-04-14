import os
import pandas as pd

def clean_data(df, output_name="clean_listings.csv"):
    print("Iniciando limpieza de datos...")

    # 1. Eliminar columnas 100% nulas
    null_cols = df.columns[df.isnull().all()]
    print(f"Columnas eliminadas (100% nulas): {list(null_cols)}")
    df = df.drop(columns=null_cols)

    # 2. Eliminar duplicados
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Duplicados eliminados: {before - after}")

    # 3. Convertir variables
    if "price" in df.columns:
        df["price"] = df["price"].replace('[\$,]', '', regex=True)
        df["price"] = pd.to_numeric(df["price"], errors='coerce')

    percent_cols = ["host_response_rate", "host_acceptance_rate"]
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].str.replace('%', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ==============================
    # 4. GUARDAR DATA LIMPIA
    # ==============================
    output_path = os.path.join("./data", output_name)
    df.to_csv(output_path, index=False)

    print(f"Archivo limpio guardado en: {output_path}")

    return df