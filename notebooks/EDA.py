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
    import polars as pl
    import numpy as np

    #
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go

    #
    import sklearn
    return np, pl, px, sklearn


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

        df_gaps = (
            df_train
            .sort(["Place_ID", "Date"])
            .with_columns(
                date_diff = pl.col("Date")
                    .diff()
                    .over("Place_ID")
            )
        ).select(['Place_ID','Date','date_diff'])

        # continuity_check = (
        #     df_gaps
        #     .group_by("Place_ID")
        #     .agg(
        #         is_continuous = (pl.col("date_diff") <= pl.duration(days=1))
        #             .all()
        #     )
        # )

        gaps = df_gaps.filter(pl.col("date_diff") > pl.duration(days=1)).select(['Place_ID','Date','date_diff'])

        return gaps

    _()
    return


@app.cell
def _(df_train, pl):
    df_train.filter(pl.col('Place_ID')=='T5P5MTS')
    return


@app.cell
def _(df_train, pl):
    def _():
        df_counts = df_train.group_by('Place_ID').agg(pl.len().alias('count'))

        percentiles = df_counts.select(
            pl.col('count').min().alias('min'),
            *[
                pl.col('count').quantile(p).alias(f'percentile_{int(p*100)}')
                for p in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
            ],
            pl.col('count').max().alias('max')
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
def _(df_test, pl):
    def _():
        df_counts = df_test.group_by('Place_ID').agg(pl.len().alias('count'))
        return df_counts

    _()
    return


@app.cell
def _(df_test, pl):
    def _():
        df_counts = df_test.group_by('Place_ID').agg(pl.len().alias('count'))

        percentiles = df_counts.select(
            pl.col('count').min().alias('min'),
            *[pl.col('count').quantile(p).alias(f'percentile_{int(p*100)}') 
              for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]],
            pl.col('count').max().alias('max')
        )
        return percentiles

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Los últimos dos puntos también se cumplan en el conjunto test
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
def _(df_train, px):
    _fig = px.histogram(df_train, x="target", nbins=100)
    _fig.show()
    return


@app.cell
def _(df_train, px):
    _fig = px.histogram(df_train, x="target", nbins=100, log_y=True)
    _fig.show()
    return


@app.cell
def _(df_train, pl, px):
    _df_train = df_train.select(
        pl.col('target').log(base=2).alias('log_target')
    )
    _df_train

    _fig = px.histogram(_df_train, x="log_target", nbins=100)
    _fig.show()
    return


@app.cell
def _(pl, sklearn):
    def run_tsne(
        X: pl.DataFrame,
        num_dim: int = 2,
        max_iter: int = 300
    ) -> pl.DataFrame:

        tsne = sklearn.manifold.TSNE(
            n_components=num_dim,
            max_iter=max_iter
        )
        X_tsne = tsne.fit_transform(X)
        columns = {f"t-SNE{i+1}": X_tsne[:, i] for i in range(num_dim)}
        return pl.DataFrame(columns)
    return (run_tsne,)


@app.cell
def _():
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor, IterativeImputer


@app.cell
def _(HistGradientBoostingRegressor, IterativeImputer, df_train, pl):
    df_X_train = df_train[:,7:74]
    df_X_train_norm = df_X_train.with_columns([
        ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
        for col in df_X_train.columns
    ])

    imputer = IterativeImputer(
        estimator=HistGradientBoostingRegressor(),
        max_iter=5,
        random_state=42,
        verbose=2
    )
    df_X_train_imputed_np = imputer.fit_transform(df_X_train_norm) 

    df_X_train_imputed = pl.DataFrame(
        df_X_train_imputed_np,
        schema=df_X_train_norm.schema 
    )
    return (df_X_train_imputed,)


@app.cell
def _(df_train):
    df_train
    return


@app.cell
def _(df_X_train_imputed, run_tsne):
    df_X_train_tsne = run_tsne(df_X_train_imputed)
    return (df_X_train_tsne,)


@app.cell
def _(Optional, np, pl, px):
    def plot_reduced_data(
        X_reduced: pl.DataFrame,
        y: Optional[np.ndarray] = None, 
        title: Optional[str] = None,
        width: int = 1000,
        height: int = 700
    ):
        """
        Grafica los datos de dimensionalidad reducida (2D o 3D) usando Plotly.
        """

        # 
        columns = X_reduced.columns
        num_dim = len(columns)

        if num_dim < 2 or num_dim > 3:
            raise ValueError(f"La dimensionalidad debe ser 2 o 3. Se detectaron {num_dim} dimensiones.")

        # 2. Preparar DataFrame y configuración de color
        df_plot = X_reduced
        kwargs = {}

        if y is not None:
            # Añadir etiquetas de color
            if isinstance(y, pl.Series):
                 df_plot = df_plot.with_columns(y.alias("label"))
            else:
                 df_plot = df_plot.with_columns(pl.Series("label", y))

            # Configuración de color para Plotly
            kwargs["color"] = "label"
            kwargs["color_continuous_scale"] = px.colors.sequential.Viridis
        else:
            # Configuración de color sin etiquetas (solo para 2D, Plotly 3D no usa discrete_sequence)
            if num_dim == 2:
                kwargs["color_discrete_sequence"] = ["steelblue"]

        # 3. Construcción de la figura (2D vs 3D)
        if num_dim == 3:
            # Gráfico 3D 
            fig = px.scatter_3d(
                df_plot.to_dict(as_series=False),
                x=columns[0],
                y=columns[1],
                z=columns[2],
                title=title,
                width=width,
                height=height,
                **kwargs
            )

            # Actualizar ejes 3D
            fig.update_layout(
                scene=dict(
                    xaxis_title=columns[0],
                    yaxis_title=columns[1],
                    zaxis_title=columns[2]
                )
            )
        else: # num_dim == 2
            # Gráfico 2D
            fig = px.scatter(
                df_plot,
                x=columns[0],
                y=columns[1],
                title=title,
                width=width,
                height=height,
                **kwargs
            )

            # Actualizar ejes 2D
            fig.update_layout(
                xaxis_title=columns[0],
                yaxis_title=columns[1]
            )

        # Ajuste de tamaño de marcador
        fig.update_traces(marker=dict(size=5))

        fig.show()
    return (plot_reduced_data,)


@app.cell
def _(df_X_train_tsne, df_train, plot_reduced_data):
    plot_reduced_data(df_X_train_tsne, df_train['target'].log1p())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
