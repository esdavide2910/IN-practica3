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
    return pl, px


@app.cell
def _(mo):
    mo.md(r"""
    # 1. Carga de datos
    """)
    return


@app.cell
def _(datasets_dir, mo, pl):
    df_train_raw = pl.read_csv(datasets_dir/"train.csv")
    mo.ui.table(df_train_raw, max_columns=82)
    return (df_train_raw,)


@app.cell
def _(datasets_dir, mo, pl):
    df_test_raw = pl.read_csv(datasets_dir/"test.csv")
    mo.ui.table(df_test_raw, max_columns=77)
    return (df_test_raw,)


@app.cell
def _(mo):
    mo.md(r"""
    # 2. Preprocesado
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
    return (df_test,)


@app.cell
def _(mo):
    mo.md(r"""
    # 3. Análisis Exploratorio de Datos
    """)
    return


@app.cell
def _(df_train, pl):
    df_train.group_by('Place_ID').agg(pl.len())
    return


@app.cell
def _(df_train, pl, px):
    df_tmp1 = df_train.group_by('Date').agg(pl.len()).sort('Date')
    fig1 = px.line( df_tmp1 , x='Date', y='len')
    fig1.show()
    return


@app.cell
def _(df_test, df_train):
    (df_train.select('Place_ID').join(
        df_test.select('Place_ID'), 
        on='Place_ID', 
        how='inner')
    ).unique()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
