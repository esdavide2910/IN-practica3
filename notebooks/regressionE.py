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
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from pytorch_forecasting.models import TemporalFusionTransformer
    import lightning.pytorch as lp

    #
    import optuna
    return KFold, np, optuna, pl, root_mean_squared_error


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
    ## 2.2. División de datos en entrenamiento y test
    """)
    return


@app.cell
def _(df_test, df_train):
    #
    df_X_train = df_train.drop(['Place_ID X Date', 'target_min','target_max','target_variance','target_count'])
    #
    df_X_test = df_test.drop(['Place_ID X Date'])
    return df_X_test, df_X_train


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.3. Normalización
    """)
    return


@app.cell
def _(df_X_train, pl):
    numerical_features = (
        df_X_train.select(pl.selectors.numeric().exclude(["target"])).columns
    )

    df_X_train_norm = df_X_train.with_columns(
        [
            ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c)
            for c in numerical_features
        ]
    )
    df_X_train_norm
    return df_X_train_norm, numerical_features


@app.cell
def _(df_X_test, numerical_features, pl):
    df_X_test_norm = df_X_test.with_columns(
        [
            ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c)
            for c in numerical_features
        ]
    )
    _cols = df_X_test_norm.columns
    _cols.insert(2, pl.lit(None).cast(pl.Int16).alias('target'))
    df_X_test_norm = df_X_test_norm.select(_cols) 
    df_X_test_norm
    return (df_X_test_norm,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.4.
    """)
    return


@app.cell
def _(df_X_train_norm, pl):
    global_start_date = df_X_train_norm["Date"].min()

    df_X_train_with_time = df_X_train_norm.with_columns(
        ((pl.col("Date") - global_start_date).dt.total_days()).cast(pl.Int64).alias("time_idx")
    )
    _cols = ["time_idx"] + [c for c in df_X_train_with_time.columns if c != "time_idx"]
    df_X_train_with_time = df_X_train_with_time[_cols].drop('Date')
    df_X_train_with_time
    return df_X_train_with_time, global_start_date


@app.cell
def _(df_X_test_norm, global_start_date, pl):
    df_X_test_with_time = df_X_test_norm.with_columns(
        ((pl.col("Date") - global_start_date).dt.total_days()).cast(pl.Int64).alias("time_idx")
    )
    _cols = ["time_idx"] + [c for c in df_X_test_with_time.columns if c != "time_idx"]
    df_X_test_with_time = df_X_test_with_time[_cols].drop('Date')
    df_X_test_with_time
    return (df_X_test_with_time,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.5. Data Expansion
    """)
    return


@app.cell
def _(df_X_test_with_time, df_X_train_with_time):
    def expand_data(df):

        all_time_idx = df['time_idx'].unique()
        all_places = df['Place_ID'].unique()
        skeleton = all_time_idx.to_frame().join(all_places.to_frame(), how="cross")

        df_expanded = skeleton.join(df, on=["time_idx", "Place_ID"], how="left").sort(["Place_ID", "time_idx"])

        return df_expanded

    df_X_train_expanded = expand_data(df_X_train_with_time)
    df_X_test_expanded = expand_data(df_X_test_with_time)
    return df_X_test_expanded, df_X_train_expanded


@app.cell
def _(df_X_train_expanded):
    df_X_train_expanded
    return


@app.cell
def _(df_X_test_expanded):
    df_X_test_expanded
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.6. Null Mask Fill
    """)
    return


@app.cell
def _(df_X_test_expanded, df_X_train_expanded, numerical_features, pl):
    numerical_cols = ['target'] + numerical_features

    def apply_null_mask_and_fill(df: pl.DataFrame, cols: list) -> pl.DataFrame:
        # Crear las máscaras: 1.0 si no es nulo, 0.0 si es nulo
        mask_exprs = [
            (pl.col(c).is_not_null().cast(pl.Float32)).alias(f"{c}_mask")
            for c in cols
        ]
    
        df_transformed = df.with_columns(mask_exprs)
    
        # Rellenar los nulos originales con 0.0
        return df_transformed.fill_null(0.0)


    df_X_train_null_mask = apply_null_mask_and_fill(df_X_train_expanded, numerical_cols)
    df_X_test_null_mask = apply_null_mask_and_fill(df_X_test_expanded, numerical_cols)
    return df_X_test_null_mask, df_X_train_null_mask


@app.cell
def _(df_X_train_null_mask):
    df_X_train_null_mask
    return


@app.cell
def _(df_X_test_null_mask):
    df_X_test_null_mask
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.7. Unification
    """)
    return


@app.cell
def _(df_X_test_null_mask, df_X_train_null_mask, pl):
    df_final = pl.concat(
        [
            df_X_train_null_mask,
            df_X_test_null_mask,
        ],
        how="vertical",
    )
    df_final
    return (df_final,)


@app.cell
def _(mo):
    mo.md(r"""
    # 3. ...
    """)
    return


@app.cell
def _(df_final):
    from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer

    # Definir las columnas de máscara generadas por tu transformer
    mask_columns = [c for c in df_final.columns if c.endswith("_mask")]

    max_prediction_length = 7 # Cuántos días quieres predecir
    max_encoder_length = 21    # Cuánto historial mirar

    training = TimeSeriesDataSet(
        df_final,
        time_idx="time_idx",
        target="target",
        group_ids=["Place_ID"],
        min_encoder_length=max_encoder_length // 2, 
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["Place_ID"],
        # IMPORTANTE: Las máscaras son "Known" porque sabes qué días fallan en test
        time_varying_known_reals=["time_idx"] + mask_columns,
        # El target y las x_i son "Unknown" porque dependen del momento
        time_varying_unknown_reals=["target"] + [f"x_{i}" for i in range(1, 75)],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_gradient_stats=True,
    )

    # Crear dataloaders para PyTorch
    batch_size = 94
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    return


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
def _(X_train_full, objective, optuna, y_train_full):
    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: objective(trial, X_train_full, y_train_full),
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
