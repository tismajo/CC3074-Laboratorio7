import pandas as pd

def create_price_categories(df):
    print("Creando variable categórica de precio...")

    # ESTO ES CRÍTICO
    df = df.dropna(subset=["price"]).copy()

    q1 = df['price'].quantile(0.33)
    q2 = df['price'].quantile(0.66)

    def categorize_price(price):
        if price <= q1:
            return "Economica"
        elif price <= q2:
            return "Intermedia"
        else:
            return "Cara"

    df["price_category"] = df["price"].apply(categorize_price)

    return df


def create_dummies(df):
    print("Creando variables dicotómicas...")

    df["is_economica"] = (df["price_category"] == "Economica").astype(int)
    df["is_intermedia"] = (df["price_category"] == "Intermedia").astype(int)
    df["is_cara"] = (df["price_category"] == "Cara").astype(int)

    print("Variables dicotómicas creadas.")

    return df


def feature_engineering(df, output_name="final_dataset.csv"):
    df = create_price_categories(df)
    df = create_dummies(df)

    # Guardar dataset final
    df.to_csv(f"./data/{output_name}", index=False)
    print(f"Dataset final guardado en ./data/{output_name}")

    return df