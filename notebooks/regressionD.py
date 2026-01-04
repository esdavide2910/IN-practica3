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
    return (datasets_dir,)


@app.cell
def _():
    #
    import time
    import math

    #
    import polars as pl
    import numpy as np

    #
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go

    #
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import StratifiedKFold

    from sklearn.metrics import root_mean_squared_error

    #
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    #
    import warnings
    warnings.filterwarnings("ignore")
    return StratifiedKFold, np, optuna, pl, plt, root_mean_squared_error, torch


@app.cell
def _(torch):
    # Comprobar si CUDA está disponible
    print(f"CUDA disponible: {torch.cuda.is_available()}")

    # Ver el número de GPUs disponibles
    print(f"Número de GPUs: {torch.cuda.device_count()}")

    # Ver el nombre de la GPU actual (si hay alguna)
    if torch.cuda.is_available():
        print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")
        print(f"ID de la GPU actual: {torch.cuda.current_device()}")
    return


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
    ## 2.2. Nueva característica: día de la semana
    """)
    return


@app.cell
def _(df_train, numerical_features_cols, pl):
    weekdays_train = df_train.select(pl.col('Date').dt.weekday()).to_dummies()
    df_X_train_1 = pl.concat([weekdays_train,df_train[numerical_features_cols]], how="horizontal")
    df_X_train_1
    return (df_X_train_1,)


@app.cell
def _(df_test, numerical_features_cols, pl):
    weekdays_test = df_test.select(pl.col('Date').dt.weekday()).to_dummies()
    df_X_test_1 = pl.concat([weekdays_test,df_test[numerical_features_cols]], how="horizontal")
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
    ## 2.3. Valores faltantes
    """)
    return


@app.cell
def _():
    # from sklearn.experimental import enable_iterative_imputer
    # from sklearn.impute import IterativeImputer
    # from sklearn.ensemble import HistGradientBoostingRegressor
    # from sklearn.compose import ColumnTransformer

    # # Columnas a eliminar (todas las que contienen CH4)
    # _cols_to_drop = [c for c in numerical_features_cols if "CH4" in c]

    # # Columnas a imputar (numéricas sin CH4)
    # _cols_to_impute = [c for c in numerical_features_cols if "CH4" not in c]

    # # Imputador iterativo
    # imputer = IterativeImputer(
    #     estimator=HistGradientBoostingRegressor(),
    #     max_iter=5,
    #     random_state=42,
    #     verbose=2
    # )
    # imputer.set_output(transform="polars")

    # # ColumnTransformer solo con las columnas a imputar
    # _ct = ColumnTransformer(
    #     [("imp", imputer, _cols_to_impute)], 
    #     remainder="passthrough", 
    #     verbose_feature_names_out=False
    # )

    # _ct.set_output(transform="polars")

    # # Fit/transform sobre train y eliminar CH4
    # df_X_train_2 = _ct.fit_transform(df_X_train_1).drop(_cols_to_drop)

    # # Transform sobre test y eliminar CH4
    # df_X_test_2 = _ct.transform(df_X_test_1).drop(_cols_to_drop)
    return


@app.cell
def _(df_X_test_1, df_X_train_1, numerical_features_cols):
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer, SimpleImputer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.compose import ColumnTransformer

    # 
    _cols_not_ch4 = [c for c in numerical_features_cols if "CH4" not in c]
    _cols_ch4 = [c for c in numerical_features_cols if "CH4" in c]
    _cols_rest = [c for c in df_X_train_1.columns if c not in _cols_not_ch4+_cols_ch4]

    # IterativeImputer para columnas numéricas sin CH4
    iter_imp = IterativeImputer(
        estimator=HistGradientBoostingRegressor(),
        max_iter=5,
        random_state=42,
        verbose=2
    )
    iter_imp.set_output(transform="polars")

    # SimpleImputer + indicador para columnas CH4
    simple_imp = SimpleImputer(strategy="median", add_indicator=True)
    simple_imp.set_output(transform="polars")

    #
    def _identity_order(X):
        return X

    # ColumnTransformer combinando ambos imputers
    _ct = ColumnTransformer(
        transformers=[
            ("pass", FunctionTransformer(_identity_order), _cols_rest),
            ("iter_imp", iter_imp, _cols_not_ch4),
            ("simple_imp", simple_imp, _cols_ch4)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    # Usar salida Polars para ColumnTransformer
    _ct.set_output(transform="polars")

    #
    df_X_train_2 = _ct.fit_transform(df_X_train_1)
    #
    df_X_test_2 = _ct.transform(df_X_test_1)
    return df_X_test_2, df_X_train_2


@app.cell
def _(df_X_train_2):
    df_X_train_2[:,65:]
    return


@app.cell
def _(df_X_train_2):
    df_X_train_2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.4. Normalización
    """)
    return


@app.cell
def _(df_X_test_2, df_X_train_2, numerical_features_cols, pl):
    # Crear un diccionario con mean y std solo del entrenamiento
    scalers = {}
    for col in numerical_features_cols:
        mean = df_X_train_2[col].mean()
        std = df_X_train_2[col].std()
        scalers[col] = (mean, std)

    # Normalizar el training set
    df_X_train_3 = df_X_train_2.with_columns([
        ((pl.col(col) - scalers[col][0]) / scalers[col][1]).alias(col)
        for col in numerical_features_cols
    ])

    # Normalizar el test set usando los parámetros del entrenamiento
    df_X_test_3 = df_X_test_2.with_columns([
        ((pl.col(col) - scalers[col][0]) / scalers[col][1]).alias(col)
        for col in numerical_features_cols
    ])
    return df_X_test_3, df_X_train_3


@app.cell
def _(df_X_train_3):
    df_X_train_3
    return


@app.cell
def _(df_X_test_3):
    df_X_test_3
    return


@app.cell
def _():
    # def apply_null_mask_and_fill(df: pl.DataFrame, cols: list) -> pl.DataFrame:

    #     # Crear las máscaras: 1.0 si no es nulo, 0.0 si es nulo
    #     mask_exprs = [
    #         (pl.col(c).is_not_null().cast(pl.Float32)).alias(f"{c}_mask")
    #         for c in cols
    #     ]

    #     df_transformed = df.with_columns(mask_exprs)

    #     # Rellenar los nulos originales con 0.0
    #     return df_transformed.fill_null(0.0)


    # #
    # null_cols = [
    #     col
    #     for col in df_X_train.columns
    #     if df_X_train.select(pl.col(col).null_count()).item() > 0
    #     or df_X_test.select(pl.col(col).null_count()).item() > 0
    # ]

    # #
    # df_X_train_null_mask = apply_null_mask_and_fill(df_X_train, null_cols)
    # df_X_test_null_mask = apply_null_mask_and_fill(df_X_test, null_cols)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.5. ...
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
    n_bins = 10
    quantiles = np.quantile(y_train, q=np.linspace(0, 1, n_bins + 1))
    y_train_binned = np.digitize(y_train, quantiles[1:-1])
    return (y_train_binned,)


@app.cell
def _(mo):
    mo.md(r"""
    # 3. Modelos entrenados
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.1. Modelo LigthGBM
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
    import lightgbm as lgb

    FIXED_PARAMS_LGBM = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
    }

    def objective_LGBM(trial, X, y, y_binned):

        tuned_params = {
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'max_depth': trial.suggest_int('max_depth',3,12), 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        param = {**FIXED_PARAMS_LGBM, **tuned_params}

        skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
        rmse_scores = []

        for train_idx, val_idx in skf.split(X, y_binned):
            X_t, X_v = X[train_idx], X[val_idx]
            y_t, y_v = y[train_idx], y[val_idx]

            model = lgb.LGBMRegressor(**param)
            model.fit(
                X_t, y_t,
                eval_set=[(X_v, y_v)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=0), lgb.log_evaluation(0)]
            )

            preds = model.predict(X_v)
            rmse = root_mean_squared_error(y_v, preds)
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)


    study_LGBM = optuna.create_study(direction='minimize')
    study_LGBM.optimize(
        lambda trial: objective_LGBM(trial, X_train, y_train, y_train_binned), 
        n_trials=20,
        show_progress_bar=True
    )
    return FIXED_PARAMS_LGBM, lgb, study_LGBM


@app.cell
def _(FIXED_PARAMS_LGBM, study_LGBM):
    print("Mejores hiperparámetros:", study_LGBM.best_params)
    print("Mejor RMSE:", study_LGBM.best_value)
    best_params_LGBM = {**FIXED_PARAMS_LGBM, **study_LGBM.best_params}
    return (best_params_LGBM,)


@app.cell
def _(X_test, X_train, best_params_LGBM, df_test, lgb, pl, y_train):
    model_LGBM = lgb.LGBMRegressor(**best_params_LGBM)
    model_LGBM.fit(X_train, y_train)

    _test_preds = model_LGBM.predict(X_test)

    results_test_LGBM = df_test.select(
        pl.col('Place_ID X Date'),
        pl.lit(_test_preds).alias('target')
    )
    # results_test_LGBM.write_csv("submission/submission_12.csv")
    results_test_LGBM
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Análisis del error
    """)
    return


@app.cell
def _(
    StratifiedKFold,
    X_train,
    best_params_LGBM,
    lgb,
    np,
    plt,
    y_train,
    y_train_binned,
):
    def _():
    
        skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

        y_true = y_train
        y_pred = np.zeros(len(y_true))
    
        for train_idx, val_idx in skf.split(X_train, y_train_binned):
            X_t, X_v = X_train[train_idx], X_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]
    
            model_LGBM = lgb.LGBMRegressor(**best_params_LGBM)
            model_LGBM.fit(
                X_t, y_t,
                eval_set=[(X_v, y_v)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=0), lgb.log_evaluation(0)]
            )
    
            y_pred[val_idx] = model_LGBM.predict(X_v)
    
            error = y_pred - y_true
    
        abs_error = np.abs(error)
        rel_error = abs_error / (y_true + 1e-6)
    
        plt.scatter(y_true, y_pred, alpha=0.3)
        return plt.plot([0, max(y_train)], [0, max(y_train)], 'r--')

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.2. Modelo XGBoost
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
        "n_jobs": -1,
        "early_stopping": 30,
    }

    def objective_XGB(trial, X, y, y_binned):

        tuned_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 1200),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 1, 5),
        }

        param = {**FIXED_PARAMS_XGB, **tuned_params}

        skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
        rmse_scores = []

        for train_idx, val_idx in skf.split(X, y_binned):
            X_t, X_v = X[train_idx], X[val_idx]
            y_t, y_v = y[train_idx], y[val_idx]

            model = xgb.XGBRegressor(**param)
            model.fit(
                X_t, y_t,
                eval_set=[(X_v, y_v)],
                verbose=False
            )

            preds = model.predict(X_v)
            rmse = root_mean_squared_error(y_v, preds)
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)


    study_XGB = optuna.create_study(direction='minimize')
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
def _(X_test, X_train, best_params_XGB, df_test, pl, xgb, y_train):
    model_XGB = xgb.XGBRegressor(**best_params_XGB)
    model_XGB.fit(X_train, y_train)

    _test_preds = model_XGB.predict(X_test)

    results_test_XGB = df_test.select(
        pl.col('Place_ID X Date'),
        pl.lit(_test_preds).alias('target')
    )
    results_test_XGB.write_csv("submission/submission_14.csv")
    results_test_XGB
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Análisis del error
    """)
    return


@app.cell
def _(
    StratifiedKFold,
    X_train,
    best_params_XGB,
    np,
    plt,
    xgb,
    y_train,
    y_train_binned,
):
    def _():
    
        skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

        y_true = y_train
        y_pred = np.zeros(len(y_true))
    
        for train_idx, val_idx in skf.split(X_train, y_train_binned):
            X_t, X_v = X_train[train_idx], X_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]
    
            model_XGB = xgb.XGBRegressor(**best_params_XGB)
            model_XGB.fit(
                X_t, y_t,
                eval_set=[(X_v, y_v)],
                verbose=False
            )
    
            y_pred[val_idx] = model_XGB.predict(X_v)
    
            error = y_pred - y_true
    
        abs_error = np.abs(error)
        rel_error = abs_error / (y_true + 1e-6)
    
        plt.scatter(y_true, y_pred, alpha=0.3)
        return plt.plot([0, max(y_train)], [0, max(y_train)], 'r--')

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
