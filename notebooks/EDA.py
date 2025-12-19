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
    import sklearn
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
    mo.ui.table(df_train_raw, max_columns=None)
    return (df_train_raw,)


@app.cell
def _(datasets_dir, mo, pl):
    df_test_raw = pl.read_csv(datasets_dir/"test.csv")
    mo.ui.table(df_test_raw, max_columns=None)
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
def _(mo):
    mo.md(r"""
    # 3. Análisis exploratorio de datos
    """)
    return


@app.cell
def _(df_train):
    df_train.columns
    return


@app.cell
def _(df_train, pl):
    def _(): 
        df_train_null_count = (
            df_train.null_count()
            .transpose(include_header=True, header_name="Columna")
            .rename({'column_0':"Valores nulos"})
            .filter(pl.col('Valores nulos')>0)
        )

        return df_train_null_count

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    - 68 de las 74 características consideradas tienen valores nulos.
    - De ellas:
      - 35 son nulas en menos del 7% de los ejemplos,
      - 26 son nulas en más del 7% y menos del 28% de los ejemplos, y
      - 7 son nulas en más del 80% de los ejemplos.
    """)
    return


@app.cell
def _(df_test, df_train, pl, px):
    def _():
        train_counts = df_train.select(
            pl.col('Date').alias('Date'),
            pl.lit(1).alias('in_train')
        ).group_by('Date').agg(
            pl.col('in_train').count().alias('Train')
        )

        test_counts = df_test.select(
            pl.col('Date').alias('Date'),
            pl.lit(1).alias('in_test')
        ).group_by('Date').agg(
            pl.col('in_test').count().alias('Test')
        )

        # Full join (unión) para incluir todas las fechas 
        df_tmp = train_counts.join(
            test_counts, 
            on='Date', 
            how='full'
        )

        #
        df_tmp = df_tmp.sort('Date')

        fig = px.line(df_tmp, x='Date', y=['Train', 'Test'])
        fig.update_xaxes(
            title_text="Date"
        )
        fig.update_yaxes(
            range=[0, None],
            title_text="Number of occurrences"
        )
        fig.show()

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    - El rango de fechas del dataset (tanto entrenamiento como test) va del 2 de enero al 4 de abril (total del 94 días).
    - Tanto el conjunto de datos de entrenamiento como el de test tienen la misma proporción de ejemplos de cada fecha.
    """)
    return


@app.cell
def _(df_train, pl):
    def _():
        df_counts = df_train.group_by('Place_ID').agg(pl.len().alias('count'))
        return df_counts

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    - En el conjunto de entrenamiento hay 340 localizaciones distintas.
    """)
    return


@app.cell
def _(df_train, pl):
    def _():
        df_counts = df_train.group_by('Place_ID').agg(pl.len().alias('count'))

        percentiles = df_counts.select(
            *[pl.col('count').quantile(p).alias(f'percentile_{int(p*100)}') 
              for p in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]]
        )
        return percentiles

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    - La mayoría de localizaciones (50% de ellas) tiene registros en cada uno de los 94 días.
    - Más del 95% de localizaciones tienen al menos 69 registros.
    """)
    return


@app.cell
def _(df_test, df_train):
    (df_train.select('Place_ID').join(
        df_test.select('Place_ID'), 
        on='Place_ID', 
        how='inner')
    ).unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Ningún par de ejemplos del conjunto de entrenamiento y de test, respectivamente, coinciden en localización.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(df_train, pl, px):
    _df_train = df_train.with_columns(
        pl.col('target').log1p().alias('target_log')
    )
    _df_train['target','target_log']
    _fig = px.histogram(_df_train, x="target_log", nbins=100, log_y=True)
    _fig.show()
    return


app._unparsable_cell(
    r"""
        df_train.columns
    """,
    name="_"
)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
