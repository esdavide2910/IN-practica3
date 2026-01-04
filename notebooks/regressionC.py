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
    import warnings
    warnings.filterwarnings("ignore")

    #
    import polars as pl
    import polars.selectors as cs
    import numpy as np

    #
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go

    #
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import StratifiedKFold
    from sklearn.decomposition import PCA
    import xgboost as xgb
    from sklearn.metrics import root_mean_squared_error

    #
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return StratifiedKFold, np, optuna, pl, plt, root_mean_squared_error, xgb


@app.cell
def _(mo):
    mo.md(r"""
    # 1. Carga de datos
    """)
    return


@app.cell
def _(datasets_dir, pl):
    df_train_raw = pl.read_csv(datasets_dir/"train.csv")
    df_train_raw
    return (df_train_raw,)


@app.cell
def _(datasets_dir, pl):
    df_test_raw = pl.read_csv(datasets_dir/"test.csv")
    df_test_raw
    return (df_test_raw,)


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
    df_train = df_train_raw.select([
        pl.col(col).cast(casting[col]) if col in casting else pl.col(col).cast(pl.Float32)
        for col in df_train_raw.columns
    ])
    df_train
    return (df_train,)


@app.cell
def _(casting, df_test_raw, pl):
    df_test = df_test_raw.select([
        pl.col(col).cast(casting[col]) if col in casting else pl.col(col).cast(pl.Float32)
        for col in df_test_raw.columns
    ])
    df_test
    return (df_test,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.2. División de datos en características y valores objetivo
    """)
    return


@app.cell
def _(df_test, df_train):
    df_X_train = df_train.drop(['Place_ID','Date', 'Place_ID X Date', 
                             'target','target_min','target_max','target_variance','target_count'])

    df_y_train = df_train['target']

    df_X_test = df_test.drop(['Place_ID X Date','Place_ID','Date'])
    return df_X_test, df_X_train, df_y_train


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.3. Normalización
    """)
    return


@app.cell
def _(df_X_train, pl):
    df_X_train_norm = df_X_train.with_columns([
        ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
        for col in df_X_train.columns
    ])
    df_X_train_norm
    return (df_X_train_norm,)


@app.cell
def _(df_X_test, pl):
    df_X_test_norm = df_X_test.with_columns([
        ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
        for col in df_X_test.columns
    ])
    df_X_test_norm
    return (df_X_test_norm,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.4.
    """)
    return


@app.cell
def _(df_X_train_norm, np):
    X_train = np.array(df_X_train_norm)
    X_train
    return (X_train,)


@app.cell
def _(df_X_test_norm, np):
    X_test = np.array(df_X_test_norm)
    X_test
    return (X_test,)


@app.cell
def _(df_y_train, np):
    y_train = np.array(df_y_train)
    y_train
    return (y_train,)


@app.cell
def _(np, y_train):
    y_train_log = np.log1p(y_train)
    y_train_log
    return (y_train_log,)


@app.cell
def _():
    # class NullMaskTransformer:
    #     def __init__(self):
    #         self.indices_with_null = None

    #     def fit(self, X):
    #         # Identifica qué columnas tienen nulos solo en el set de FIT (train)
    #         nulls_per_col = np.isnan(X).any(axis=0)
    #         self.indices_with_null = np.where(nulls_per_col)[0]
    #         return self

    #     def transform(self, X):
    #         # Si no hay columnas con nulos, devuelve X con ceros
    #         if len(self.indices_with_null) == 0:
    #             return np.nan_to_num(X, nan=0.0)

    #         # Crea la máscara usando los índices guardados en el fit
    #         X_nulls = X[:, self.indices_with_null]
    #         # 1.0 si existe el dato, 0.0 si es nulo
    #         nulls_mask = (~np.isnan(X_nulls)).astype(np.float32)

    #         # Limpia originales
    #         X_clean = np.nan_to_num(X, nan=0.0)

    #         return np.concatenate([X_clean, nulls_mask], axis=1)

    #     def fit_transform(self, X):
    #         return self.fit(X).transform(X)
    return


@app.cell
def _():
    # _null_mask_transformer = NullMaskTransformer()
    # X_train = _null_mask_transformer.fit_transform(df_X_train_norm)
    # X_test = _null_mask_transformer.transform(df_X_test_norm)
    # y_train = df_y_train.to_numpy()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.5. Reducción de dimensionalidad
    """)
    return


@app.cell
def _():
    # imputer = IterativeImputer(
    #     estimator=HistGradientBoostingRegressor(),
    #     max_iter=5,
    #     random_state=42,
    #     verbose=2
    # )
    # df_X_train_imputed_np = imputer.fit_transform(df_X_train_norm) 

    # df_X_train_imputed = pl.DataFrame(
    #     df_X_train_imputed_np,
    #     schema=df_X_train_norm.schema 
    # )
    return


@app.cell
def _():
    # df_X_test_imputed_np = imputer.transform(df_X_test_norm)

    # df_X_test_imputed = pl.DataFrame(
    #     df_X_test_imputed_np,
    #     schema=df_X_test_norm.schema 
    # )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.5. Selección de características
    """)
    return


@app.cell
def _():
    # X_train = df_X_train_imputed.select(cs.contains('density')).to_numpy()
    # X_test = df_X_test_imputed.select(cs.contains('density')).to_numpy()
    # y_train = df_y_train.to_numpy()
    return


@app.cell
def _():
    # X_train = df_X_train_imputed.to_numpy()
    # X_test = df_X_test_imputed.to_numpy()
    # y_train = df_y_train.to_numpy()
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 3. Elección de hiperparámetros
    """)
    return


@app.cell
def _(StratifiedKFold, np, root_mean_squared_error, xgb):
    FIXED_PARAMS = {
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

    def objective(trial, X, y):
    
        tuned_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 1, 5),
        }

        param = {**FIXED_PARAMS, **tuned_params}

        n_bins = 15  
        quantiles = np.quantile(y, q=np.linspace(0, 1, n_bins + 1))
        y_binned = np.digitize(y, quantiles[1:-1])

        skf = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )
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
    return FIXED_PARAMS, objective


@app.cell
def _(X_train, objective, optuna, y_train):
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train), 
        n_trials=15,
        show_progress_bar=True
    )
    return (study,)


@app.cell
def _(FIXED_PARAMS, study):
    print("Mejores hiperparámetros:", study.best_params)
    print("Mejor RMSE:", study.best_value)
    best_params = {**FIXED_PARAMS, **study.best_params}
    return (best_params,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.1. Análisis del error
    """)
    return


@app.cell
def _(StratifiedKFold, X_train, best_params, np, xgb, y_train):
    oof_pred = np.zeros(len(y_train))
    oof_true = y_train.copy()

    n_bins = 10  
    quantiles = np.quantile(y_train, q=np.linspace(0, 1, n_bins + 1))
    y_binned = np.digitize(y_train, quantiles[1:-1])

    skf = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    for train_idx, val_idx in skf.split(X_train, y_binned):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]

        model = xgb.XGBRegressor(**best_params)
        model.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            verbose=False
        )

        oof_pred[val_idx] = model.predict(X_v)

        error = oof_pred - oof_true
    abs_error = np.abs(error)
    rel_error = abs_error / (oof_true + 1e-6)
    return oof_pred, oof_true


@app.cell
def _(oof_pred, oof_true, plt, y_train):
    plt.scatter(oof_true, oof_pred, alpha=0.3)
    plt.plot([0, max(y_train)], [0, max(y_train)], 'r--')
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 4. Entrenamiento e inferencia del modelo final
    """)
    return


@app.cell
def _(X_test, X_train, best_params, xgb, y_train_log):
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train_log)

    y_test = final_model.predict(X_test)
    return (y_test,)


@app.cell
def _(df_test, np, pl, y_test):
    results_test = df_test.select(
        pl.col('Place_ID X Date'),
        pl.lit(np.expm1(y_test)).alias('target')
    )
    results_test.write_csv("submission/submission_8.csv")
    results_test
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
