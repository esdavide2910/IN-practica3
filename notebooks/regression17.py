import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    from pathlib import Path

    # Get the path to the “notebook/” directory
    notebook_dir = mo.notebook_dir()

    # Get the path to the "datasets/" directory 
    datasets_dir = notebook_dir.parent / "dataset"

    # Get the path to the "src/" directory
    src_dir = notebook_dir.parent / "src"

    # Add the source directory to the search Python path
    import sys
    sys.path.append(str(src_dir))
    print(f"Versión de Python: {sys.version}")
    return (datasets_dir,)


@app.cell
def _():
    # Módulos estándar de Python para utilidades básicas
    import time
    import math

    # Bibliotecas para manejo y procesamiento de datos
    import polars as pl
    import polars.selectors as cs
    import numpy as np

    # Bibliotecas para visualización de datos
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go

    #  Módulos de scikit-learn para modelado y evaluación
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import root_mean_squared_error

    # Biblioteca para optimización de hiperparámetros mediante búsqueda automatizada
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Configuración para ignorar warnings de Python
    import warnings
    warnings.filterwarnings("ignore")
    return StratifiedKFold, np, optuna, pl, root_mean_squared_error


@app.cell
def _(mo):
    mo.md(r"""
    # 1. Carga de datos
    """)
    return


@app.cell
def _(datasets_dir, pl):
    # Carga los datos de entrenamiento
    df_train_raw = pl.read_csv(datasets_dir/"train.csv")
    df_train_raw 
    return (df_train_raw,)


@app.cell
def _(datasets_dir, pl):
    # Carga los datos de test
    df_test_raw = pl.read_csv(datasets_dir/"test.csv")
    df_test_raw
    return (df_test_raw,)


@app.cell
def _(df_test_raw, df_train_raw):
    # Identifica las columnas de características identificativas
    id_features_cols = df_test_raw.columns[0:3]
    print(f"Columnas de características identificativas: {id_features_cols}")

    # Identifica las columnas de características numéricas
    numerical_features_cols = df_test_raw.columns[3:]
    print(f"Columnas de características numéricas: {numerical_features_cols}")

    # Identifica las columnas de valores target
    target_cols = df_train_raw.columns[3:8]
    print(f"Columnas de valores objetivo: {target_cols}")
    return (numerical_features_cols,)


@app.cell
def _(mo):
    mo.md(r"""
    # 2. Preprocesado de datos
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.1. Corrección de tipos
    """)
    return


@app.cell
def _(pl):
    # Define un diccionario con nombres de columnas como key y tipo como valor
    casting = {
        "Place_ID X Date": pl.String,
        "Date": pl.Date,
        "Place_ID": pl.Categorical,
        "target": pl.Int16,
        "target_min": pl.Int16,
        "target_max": pl.Int16,
        "target_variance": pl.Float32,
        "target_count": pl.Int16
    }
    return (casting,)


@app.cell
def _(casting, df_train_raw, pl):
    # Castea cada columna al tipo especificado en el diccionario. Por defecto, castea a Float32
    df_train = df_train_raw.select([
        pl.col(col).cast(casting[col]) if col in casting else pl.col(col).cast(pl.Float32)
        for col in df_train_raw.columns
    ])
    df_train
    return (df_train,)


@app.cell
def _(casting, df_test_raw, pl):
    # Castea cada columna al tipo especificado en el diccionario. Por defecto, castea a Float32
    df_test = df_test_raw.select([
        pl.col(col).cast(casting[col]) if col in casting else pl.col(col).cast(pl.Float32)
        for col in df_test_raw.columns
    ])
    df_test
    return (df_test,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.2. Reconstrucción de la Serie Temporal
    """)
    return


@app.cell
def _(date, df_test, df_train, pl):
    # Genera el rango completo de fechas y extrae los Place_ID únicos para asegurar la continuidad temporal
    all_dates = pl.date_range(date(2020, 1, 2), date(2020, 4, 4), "1d", eager=True).alias("Date")
    all_places_train = df_train.select("Place_ID").unique()
    all_places_test = df_test.select("Place_ID").unique()
    return all_dates, all_places_test, all_places_train


@app.cell
def _(all_dates, all_places_train, df_train, pl):
    # Realiza un upsampling del set de entrenamiento: crea una fila por cada día/lugar para evitar saltos en los lags
    grid_train = all_places_train.join(pl.DataFrame(all_dates), how="cross")
    df_train_complete = grid_train.join(df_train, on=["Place_ID", "Date"], how="left").sort(["Place_ID", "Date"])
    df_train_complete
    return (df_train_complete,)


@app.cell
def _(all_dates, all_places_test, df_test, pl):
    # Realiza un upsampling del set de test: asegura que la estructura temporal sea idéntica a la de entrenamiento
    grid_test = all_places_test.join(pl.DataFrame(all_dates), how="cross")
    df_test_complete = grid_test.join(df_test, on=["Place_ID", "Date"], how="left").sort(["Place_ID", "Date"])
    df_test_complete
    return (df_test_complete,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.3. Generación de Variables de Retraso (Lag Features)
    """)
    return


@app.cell
def _(df_train_complete, numerical_features_cols, pl):
    # Definimos los intervalos de retraso (pasado)
    lags = [1, 2, 7]

    # Generamos las variables de retraso en un solo paso
    df_train_full = df_train_complete.with_columns(
        [
            # Lags: Valores de días anteriores
            pl.col(c).shift(lag).over("Place_ID").alias(f"{c}_lag{lag}")
            for c in numerical_features_cols
            for lag in lags
        ] 
    )
    df_train_full
    return df_train_full, lags


@app.cell
def _(df_test_complete, lags, numerical_features_cols, pl):
    # Genera las mismas variables de retraso en el conjunto test
    df_test_full = df_test_complete.with_columns(
        [
            pl.col(c).shift(lag).over("Place_ID").alias(f"{c}_lag{lag}")
            for c in numerical_features_cols
            for lag in lags
        ] 
    )
    df_test_full
    return (df_test_full,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.4. Normalización
    """)
    return


@app.cell
def _():
    import re

    # Función para obtener el nombre original de la variable eliminando el sufijo del lag
    # Esto permite que 'x_1_lag1' sea escalada usando los parámetros calculados para 'x_1'
    def base_col(col: str) -> str:
        return re.sub(r"_(lag)\d+$", "", col)
    return (base_col,)


@app.cell
def _(df_train_full, numerical_features_cols):
    # Calcula y almacena la media y desviación estándar de cada característica original
    # Solo se utiliza el set de entrenamiento para evitar el data leakage 
    scalers = {}
    for col in numerical_features_cols:
        mean = df_train_full[col].mean()
        std = df_train_full[col].std()
        scalers[col] = (mean, std)
    return (scalers,)


@app.cell
def _(base_col, df_train_full, pl, scalers):
    # Aplica la normalización Z-score a todas las columnas del set de entrenamiento
    # Se usa base_col para aplicar la escala de la variable raíz tanto a la original como a sus lags
    df_train_norm = df_train_full.with_columns(
        [   (
                (pl.col(col) - scalers[base_col(col)][0]) / scalers[base_col(col)][1]
            ).alias(col)
            for col in df_train_full.columns
            if base_col(col) in scalers
        ]
    )
    df_train_norm
    return (df_train_norm,)


@app.cell
def _(base_col, df_test_full, pl, scalers):
    # Aplica la normalización al set de test usando estrictamente los parámetros de entrenamiento
    df_test_norm = df_test_full.with_columns(
        [   (
                (pl.col(col) - scalers[base_col(col)][0]) / scalers[base_col(col)][1]
            ).alias(col)
            for col in df_test_full.columns
            if base_col(col) in scalers
        ]
    )
    df_test_norm
    return (df_test_norm,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.5. Alineación de Datos y Filtrado de Instancias de Entrenamiento
    """)
    return


@app.cell
def _(df_train, df_train_norm):
    # Filtra el set de entrenamiento extendido para conservar solo las filas que existían originalmente
    # Esto elimina los días "vacíos" creados durante el upsampling que no tienen valor objetivo (target)
    df_train_ready = df_train['Place_ID','Date'].join(df_train_norm, on=["Place_ID", "Date"], how="left").sort(["Place_ID", "Date"])
    return (df_train_ready,)


@app.cell
def _(df_test, df_test_norm):
    # Filtra el set de test para recuperar la estructura original de la competición
    # Mantiene las variables normalizadas y los lags para las fechas requeridas en la entrega
    df_test_ready = df_test['Place_ID','Date'].join(df_test_norm, on=["Place_ID", "Date"], how="left").sort(["Place_ID", "Date"])
    return (df_test_ready,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.6. Vectorización y Preparación de Matrices de Entrada
    """)
    return


@app.cell
def _(base_col, df_train_ready, np, numerical_features_cols):
    # Convierte las características de entrenamiento a un array de NumPy para el modelo
    # Se seleccionan tanto las variables originales como sus lags mediante la función base_col
    X_train = np.array([
        df_train_ready[c] for c in df_train_ready.columns
        if base_col(c) in numerical_features_cols
    ]).T
    return (X_train,)


@app.cell
def _(base_col, df_test_ready, np, numerical_features_cols):
    # Convierte las características de test a un array de NumPy siguiendo el mismo orden de columnas
    # Este array será la entrada final para generar las predicciones de la competición
    X_test = np.array([
        df_test_ready[c] for c in df_test_ready.columns
        if base_col(c) in numerical_features_cols
    ]).T
    return (X_test,)


@app.cell
def _(df_train_ready, np):
    y_train = np.array(df_train_ready['target'])
    y_train
    return (y_train,)


@app.cell
def _(np, y_train):
    n_bins = 100
    quantiles = np.quantile(y_train, q=np.linspace(0, 1, n_bins + 1))
    y_train_binned = np.digitize(y_train, quantiles[1:-1])
    return (y_train_binned,)


@app.cell
def _(mo):
    mo.md(r"""
    # 3. Entrenamiento e inferencia del modelo XGBoost
    """)
    return


@app.cell
def _(
    StratifiedKFold,
    X_train,
    np,
    optuna,
    root_mean_squared_error,
    y_train,
    y_train_binned,
):
    import xgboost as xgb
    xgb.set_config(verbosity=0)

    FIXED_PARAMS_XGB = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "booster": "gbtree",
        "verbosity": 0,
        "random_state": 42,
        "n_jobs": 1,
        "early_stopping": 30,
    }

    def objective_XGB(trial, X, y, y_binned):

        tuned_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.5, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 1000, 1100),
            "subsample": trial.suggest_float("subsample", 0.7, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.8),
            "max_depth": trial.suggest_int("max_depth", 7, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 12),
            "gamma": trial.suggest_float("gamma", 1, 3),
        }

        param = {**FIXED_PARAMS_XGB, **tuned_params}

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
            X_t, X_v = X[train_idx], X[val_idx]
            y_t, y_v = y[train_idx], y[val_idx]

            model = xgb.XGBRegressor(**param)
            model.fit(
                X_t, y_t,
                eval_set=[(X_v, y_v)],
                verbose=False,
            )

            preds = model.predict(X_v)
            rmse = root_mean_squared_error(y_v, preds)
            rmse_scores.append(rmse)

            trial.report(rmse, fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(rmse_scores)


    study_XGB = optuna.create_study(
        direction='minimize', 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )

    study_XGB.optimize(
        lambda trial: objective_XGB(trial, X_train, y_train, y_train_binned), 
        n_trials=20,
        show_progress_bar=True
    )
    return FIXED_PARAMS_XGB, study_XGB, xgb


@app.cell
def _(FIXED_PARAMS_XGB, study_XGB):
    best_params_XGB = {**FIXED_PARAMS_XGB, **study_XGB.best_params}
    print("Mejores hiperparámetros:", best_params_XGB)
    print("Mejor RMSE:", study_XGB.best_value)
    return (best_params_XGB,)


@app.cell
def _(X_test, X_train, best_params_XGB, df_test_ready, pl, xgb, y_train):
    model_XGB = xgb.XGBRegressor(**best_params_XGB)
    model_XGB.fit(X_train, y_train)

    _test_preds = model_XGB.predict(X_test)

    results_test_XGB = df_test_ready.select(
        pl.col('Place_ID X Date'),
        pl.lit(_test_preds).alias('target')
    )
    results_test_XGB.write_csv("submission/submission_17.csv")
    results_test_XGB
    return


if __name__ == "__main__":
    app.run()
