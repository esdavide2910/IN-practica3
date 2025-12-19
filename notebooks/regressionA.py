import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


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
    from sklearn.model_selection import KFold
    import lightgbm as lgb
    from sklearn.metrics import root_mean_squared_error

    #
    import optuna
    return (
        HistGradientBoostingRegressor,
        IterativeImputer,
        KFold,
        lgb,
        np,
        optuna,
        pl,
        root_mean_squared_error,
    )


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore")
    return


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
    #
    df_train = df_train_raw.select([
        pl.col(col).cast(casting[col]) if col in casting else pl.col(col).cast(pl.Float32)
        for col in df_train_raw.columns
    ])
    df_train
    return (df_train,)


@app.cell
def _(casting, df_test_raw, pl):
    #
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
    #
    X_train = df_train.drop(['Place_ID','Date', 'Place_ID X Date', 
                             'target','target_min','target_max','target_variance','target_count'])
    y_train = df_train['target']
    #
    X_test = df_test.drop(['Place_ID X Date','Place_ID','Date'])
    return X_test, X_train, y_train


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.3. Normalización
    """)
    return


@app.cell
def _(X_train, pl):
    X_train_norm = X_train.with_columns([
        ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
        for col in X_train.columns
    ])
    X_train_norm
    return (X_train_norm,)


@app.cell
def _(X_test, pl):
    X_test_norm = X_test.with_columns([
        ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
        for col in X_test.columns
    ])
    X_test_norm
    return (X_test_norm,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.4. Imputación de valores faltantes
    """)
    return


@app.cell
def _(HistGradientBoostingRegressor, IterativeImputer, X_train_norm, pl):
    imputer = IterativeImputer(
        estimator=HistGradientBoostingRegressor(),
        max_iter=5,
        random_state=42,
        verbose=2
    )
    X_train_imputed_np = imputer.fit_transform(X_train_norm) 

    X_train_imputed = pl.DataFrame(
        X_train_imputed_np,
        schema=X_train_norm.schema 
    )
    return X_train_imputed, imputer


@app.cell
def _(X_test_norm, imputer, pl):
    X_test_imputed_np = imputer.transform(X_test_norm)

    X_test_imputed = pl.DataFrame(
        X_test_imputed_np,
        schema=X_test_norm.schema 
    )
    return (X_test_imputed,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.5. Selección de características
    """)
    return


@app.cell
def _(X_train_imputed):
    X_train_imputed
    return


@app.cell
def _(X_test_imputed, X_train_imputed, y_train):
    X_train_final = X_train_imputed.to_numpy()
    X_test_final = X_test_imputed.to_numpy()
    y_train_final = y_train.to_numpy()
    return X_test_final, X_train_final, y_train_final


@app.cell
def _(mo):
    mo.md(r"""
    # 3. Elección de hiperparámetros
    """)
    return


@app.cell
def _(KFold, lgb, np, optuna, root_mean_squared_error):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial, X, y):
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'max_depth': trial.suggest_int('max_depth',3,15), 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []

        for train_idx, val_idx in kf.split(X):
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
    return (objective,)


@app.cell
def _(X_train_final, objective, optuna, y_train_final):
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train_final, y_train_final), 
        n_trials=50,
        show_progress_bar=True
    )
    return (study,)


@app.cell
def _(study):
    print("Mejores hiperparámetros:", study.best_params)
    print("Mejor RMSE:", study.best_value)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Entrenamiento e inferencia del modelo final
    """)
    return


@app.cell
def _(X_test_final, X_train_final, lgb, study, y_train_final):
    best_params = study.best_params

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train_final, y_train_final)

    y_test = final_model.predict(X_test_final)
    return (y_test,)


@app.cell
def _(df_test, pl, y_test):
    results_test = df_test.select(
        pl.col('Place_ID X Date'),
        pl.lit(y_test).alias('target')
    )
    results_test.write_csv("submission/submission_1.csv")
    results_test
    return


if __name__ == "__main__":
    app.run()
