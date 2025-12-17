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
    import numpy as np

    #
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go

    #
    from sklearn.model_selection import KFold
    from sklearn.metrics import root_mean_squared_error
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    return (
        ColumnTransformer,
        HistGradientBoostingRegressor,
        IterativeImputer,
        KFold,
        StandardScaler,
        np,
        pl,
    )


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
def _(df_train_raw, pl):
    #
    df_train = df_train_raw.drop('Place_ID X Date')

    #
    casting = {
        "Date": pl.Date,
        "Place_ID": pl.Categorical,
        "target": pl.Int16,
        "target_min": pl.Int16,
        "target_max": pl.Int16,
        "target_variance": pl.Float32,
        "target_count": pl.Int16
    }
    df_train = df_train.select([
        pl.col(col).cast(casting[col]) if col in casting else pl.col(col).cast(pl.Float32)
        for col in df_train.columns
    ])

    #
    df_train
    return casting, df_train


@app.cell
def _(casting, df_test_raw, pl):
    #
    df_test = df_test_raw.drop('Place_ID X Date')
    df_test = df_test.select([
        pl.col(col).cast(casting[col]) if col in casting else pl.col(col).cast(pl.Float32)
        for col in df_test.columns
    ])
    df_test
    return (df_test,)


@app.cell
def _(df_test, df_train):
    X_train = df_train[:, 7:73]
    X_test = df_test[:, 2:68]

    y_train = df_train['target']
    return X_test, X_train


@app.cell
def _(df_test, df_train):
    X_train = df_train[:, 7:73].to_numpy()
    X_test = df_test[:, 2:68].to_numpy()

    y_train = df_train['target'].to_numpy()
    return X_test, X_train


@app.cell
def _(ColumnTransformer, StandardScaler, X_test, X_train, pl):
    columns_to_scale = []
    for col_name, col_type in X_test.schema.items():
        if col_type in (pl.Float32, pl.Float64):
            columns_to_scale.append(col_name)

    preprocessor = ColumnTransformer(
        transformers=[
            ('norm', StandardScaler(), columns_to_scale)
        ],
        remainder='passthrough'  # Mantener las demás columnas
    )

    X_train_norm = preprocessor.fit_transform(X_train)
    X_test_norm = preprocessor.transform(X_test)
    return X_test_norm, X_train_norm


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.x. Imputación de valores
    """)
    return


@app.cell
def _(HistGradientBoostingRegressor, IterativeImputer, X_train_norm):
    imputer = IterativeImputer(
        estimator=HistGradientBoostingRegressor(),
        random_state=42,
        verbose=2
    )
    X_imputed_train = imputer.fit_transform(X_train_norm) 
    return (imputer,)


@app.cell
def _(X_test_norm, imputer):
    X_imputed_test = imputer.transform(X_test_norm)
    return


@app.cell
def _(KFold, X, lgb, mean_squared_error, np, y_log):
    # 2. Configuración de la Validación Cruzada
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X)) # Out-of-fold predictions
    models = []

    # Parámetros básicos de LightGBM
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'device': 'cpu' # Cambia a 'gpu' si tienes una disponible
    }

    print("Iniciando Entrenamiento con 5-Folds...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_log[train_idx], y_log[val_idx]
    
        # Crear datasets de LightGBM
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
        # Entrenar modelo
        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )
    
        # Guardar predicciones y modelo
        oof_predictions[val_idx] = model.predict(X_val)
        models.append(model)
        print(f"Fold {fold+1} completado.")

    # 3. Evaluación Final (RMSLE Global)
    # Como predijimos en log, calculamos el RMSE de los logs directamente
    rmsle_score = np.sqrt(mean_squared_error(y_log, oof_predictions))
    print(f"\n--- SCORE CV RMSLE: {rmsle_score:.5f} ---")
    return (X_train,)


@app.cell
def _(mo):
    mo.md(r"""
    # 5.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
