from ExploratoryAnalysis import run_eda
from CleanData import clean_data
from FeatureEngineering import feature_engineering
from DataSplit import create_split, load_split
from ModelTraining import (
    select_features,
    train_logistic_model,
    cross_validation,
    evaluate_model
)
import pandas as pd

df = None

while True:
    print("\n=== MENÚ ===")
    print("1. Ejecutar Exploratory Data Analysis")
    print("2. Limpiar datos")
    print("3. Crear variables para modelo")
    print("4. Crear Train/Test split")
    print("5. Cargar Train/Test split")
    print("6. Entrenar modelo de regresión logística")
    print("7. Analizar modelo")
    print("8. Tunear modelo")
    print("9. Comparar modelos")
    print("10. Salir")

    opcion = input("Selecciona una opción: ")

    if opcion == "1":
        archive = input("Ingresa el nombre del archivo a analizar: → ./data/")

        if not archive:
            print("Debes ingresar un archivo válido.")
            continue

        try:
            df = run_eda(archive)
        except FileNotFoundError:
            print("El archivo no existe. Verifica la ruta.")
        except Exception as e:
            print(f"Error al procesar el archivo: {e}")

    elif opcion == "2":
        if df is None:
            print("Primero debes cargar los datos con EDA.")
        else:
            print("Limpiando los datos...")
            df = clean_data(df)

    elif opcion == "3":
        if df is None:
            print("Primero debes cargar los datos (EDA).")
        else:
            df = feature_engineering(df)
            print(df[["price", "price_category", "is_economica", "is_intermedia", "is_cara"]].head())

    elif opcion == "4":
        if df is None:
            print("Primero debes cargar y preparar los datos.")
        else:
            if "is_cara" not in df.columns:
                print("Primero debes crear las variables (opción 3).")
            else:
                X_train, X_test, y_train, y_test = create_split(df, "is_cara")
                print("NaNs en target:", df["is_cara"].isnull().sum())
                print("Split creado correctamente.")
                print(f"Train: {X_train.shape}")
                print(f"Test: {X_test.shape}")

    elif opcion == "5":
        if df is None:
            print("Primero debes cargar y preparar los datos.")
        else:
            try:
                X_train, X_test, y_train, y_test = load_split(df, "is_cara")

                print("Split cargado correctamente.")
                print(f"Train: {X_train.shape}")
                print(f"Test: {X_test.shape}")

            except FileNotFoundError:
                print("No existe un split guardado. Primero crea uno (opción 4).")
    elif opcion == "6":
        if df is None:
            print("Primero debes cargar y preparar los datos.")
        else:
            if "is_cara" not in df.columns:
                print("Primero debes crear las variables (opción 3).")
            else:
                print("Preparando datos para el modelo...")

                # 1. Seleccionar features
                X, y = select_features(df, "is_cara")

                # 2. Crear split (usando dataset combinado)
                df_model = pd.concat([X, y], axis=1)

                X_train, X_test, y_train, y_test = create_split(df_model, "is_cara")

                # 3. Entrenar modelo
                model = train_logistic_model(X_train, y_train)

                # 4. Validación cruzada
                cross_validation(model, X_train, y_train)

                # 5. Evaluación
                evaluate_model(model, X_test, y_test)

    elif opcion == "7":
        from ModelAnalysis import (
            calculate_vif,
            correlation_analysis,
            logistic_regression_summary,
            check_overfitting,
            plot_learning_curve,
            profile_model
        )

        if df is None:
            print("Primero prepara los datos.")
        else:
            X, y = select_features(df, "is_cara")
            df_model = pd.concat([X, y], axis=1)

            X_train, X_test, y_train, y_test = load_split(df_model, "is_cara")

            model = train_logistic_model(X_train, y_train)

            # 🔍 análisis
            correlation_analysis(X_train)
            calculate_vif(X_train)

            logistic_regression_summary(X_train, y_train)

            check_overfitting(model, X_train, y_train, X_test, y_test)

            plot_learning_curve(model, X_train, y_train)

            profile_model(train_logistic_model, X_train, y_train)
    
    elif opcion == "8":
        from ModelTuning import tune_logistic

        X, y = select_features(df, "is_cara")
        df_model = pd.concat([X, y], axis=1)

        X_train, X_test, y_train, y_test = load_split(df_model, "is_cara")

        best_model = tune_logistic(X_train, y_train)

        evaluate_model(best_model, X_test, y_test)

    elif opcion == "9":
        from ModelComparison import train_all_models, compare_models, plot_results

        X, y = select_features(df, "is_cara")
        df_model = pd.concat([X, y], axis=1)

        X_train, X_test, y_train, y_test = load_split(df_model, "is_cara")

        models = train_all_models(X_train, y_train)
        results = compare_models(models, X_test, y_test)

        plot_results(results)

    elif opcion == "10":
        print("Saliendo...")
        break

    else:
        print("Opción inválida")
