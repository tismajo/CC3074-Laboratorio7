import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

def select_features(df, target_col):
    print("Seleccionando variables para el modelo...")

    selected_cols = [
        "accommodates",
        "price",
        "minimum_nights",
        "maximum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "review_scores_rating"
    ]

    df_model = df[selected_cols + [target_col]].copy()

    df_model = df_model.dropna(subset=[target_col])

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 🔥 ESCALADO
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y

def train_logistic_model(X_train, y_train):
    print("Entrenando modelo de regresión logística...")

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    return model


def cross_validation(model, X_train, y_train):
    print("Aplicando validación cruzada...")

    scores = cross_val_score(model, X_train, y_train, cv=5)

    print(f"Scores CV: {scores}")
    print(f"Promedio CV: {scores.mean()}")


def evaluate_model(model, X_test, y_test):
    print("Evaluando modelo...")

    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte:")
    print(classification_report(y_test, y_pred))