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
def _(df_train):
    df_train
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.2. Selección de características
    """)
    return


@app.cell
def _(df_train, numerical_features_cols):
    df_X_train_1 = df_train.select(
        [col for col in numerical_features_cols]
    )
    df_X_train_1
    return (df_X_train_1,)


@app.cell
def _(df_test, numerical_features_cols):
    df_X_test_1 = df_test.select(
        [col for col in numerical_features_cols]
    )
    df_X_test_1
    return (df_X_test_1,)


@app.cell
def _(df_train):
    df_y_train = df_train['target']
    df_y_train
    return (df_y_train,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.3. Imputación de valores faltantes
    """)
    return


@app.cell
def _(df_X_test_1, df_X_train_1):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import HistGradientBoostingRegressor

    imputer = IterativeImputer(
        estimator=HistGradientBoostingRegressor(),
        max_iter=5,
        random_state=42,
        verbose=2
    )
    imputer.set_output(transform="polars")

    # 
    df_X_train_2 = imputer.fit_transform(df_X_train_1)
    df_X_test_2 = imputer.transform(df_X_test_1)
    return df_X_test_2, df_X_train_2


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.4. Normalización
    """)
    return


@app.cell
def _(df_X_test_2, df_X_train_2, pl):
    # Crear un diccionario con mean y std solo del entrenamiento
    scalers = {}
    for col in df_X_train_2.columns:
        mean = df_X_train_2[col].mean()
        std = df_X_train_2[col].std()
        scalers[col] = (mean, std)

    # Normalizar el training set
    df_X_train_3 = df_X_train_2.with_columns([
        ((pl.col(col) - scalers[col][0]) / scalers[col][1]).alias(col)
        for col in df_X_train_2.columns
    ])

    # Normalizar el test set usando los parámetros del entrenamiento
    df_X_test_3 = df_X_test_2.with_columns([
        ((pl.col(col) - scalers[col][0]) / scalers[col][1]).alias(col)
        for col in df_X_test_2.columns
    ])
    return df_X_test_3, df_X_train_3


@app.cell
def _(df_X_train_3):
    df_X_train_3
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.5. Vectorización y Preparación de Matrices de Entrada
    """)
    return


@app.cell
def _(df_X_test_3, df_X_train_3, np):
    X_train = np.array(df_X_train_3)
    X_test = np.array(df_X_test_3)
    return X_test, X_train


@app.cell
def _(df_y_train, np):
    y_train = np.array(df_y_train)
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
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 800, 1500),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.8),
            "max_depth": trial.suggest_int("max_depth", 5, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 1, 20),
        }

        param = {**FIXED_PARAMS_XGB, **tuned_params}
    
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []

        # Transformamos y a escala logarítmica
        y_log = np.log1p(y)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
            X_t, X_v = X[train_idx], X[val_idx]
            y_t_log, y_v_log = y_log[train_idx], y_log[val_idx] # Entrenamiento en LOG
            y_v_original = y[val_idx] # Validación en escala ORIGINAL

            model = xgb.XGBRegressor(**param)
            model.fit(
                X_t, y_t_log,
                eval_set=[(X_v, y_v_log)], # Evalúa en escala LOG para early stopping
                verbose=False,
            )

            # Predicción y reversión del logaritmo
            preds_log = model.predict(X_v)
            preds_original = np.expm1(preds_log) # Volvemos a la escala real

            # Calculamos RMSE en la escala que te interesa (sin logaritmo)
            rmse = root_mean_squared_error(y_v_original, preds_original)
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
def _(X_test, X_train, best_params_XGB, df_test, np, pl, xgb, y_train):
    model_XGB = xgb.XGBRegressor(**best_params_XGB)
    model_XGB.fit(X_train, np.log1p(y_train))

    _test_preds_log = model_XGB.predict(X_test)
    _test_preds_original = np.expm1(_test_preds_log)

    results_test_XGB = df_test.select(
        pl.col('Place_ID X Date'),
        pl.lit(_test_preds_original).alias('target')
    )
    results_test_XGB.write_csv("submission/submission_08.csv")
    results_test_XGB
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
