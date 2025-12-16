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
    from xgboost import XGBRegressor
    from sklearn.metrics import root_mean_squared_error
    return (pl,)


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
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. División de datos
    """)
    return


@app.cell
def _(df_test, df_train):
    X_train = df_train[:, 7:].to_numpy()
    X_test = df_test[:, 2:].to_numpy()
    return


@app.cell
def _(df_train):
    y_train = df_train['target'].to_numpy()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 4. Selección del modelo
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 5.
    """)
    return


if __name__ == "__main__":
    app.run()
