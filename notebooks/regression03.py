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
    return math, np, pl, time


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
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import HistGradientBoostingRegressor

    imputer = SimpleImputer(strategy="median", add_indicator=True)
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
def _(df_X_test_3, df_X_train_3, df_y_train, np):
    X_train_full = np.array(df_X_train_3)
    y_train_full = np.array(df_y_train)

    X_test = np.array(df_X_test_3)
    return X_test, X_train_full, y_train_full


@app.cell
def _(np, y_train_full):
    n_bins = 100
    quantiles = np.quantile(y_train_full, q=np.linspace(0, 1, n_bins + 1))
    y_train_binned = np.digitize(y_train_full, quantiles[1:-1])
    return (y_train_binned,)


@app.cell
def _():
    from torch.utils.data import Dataset, DataLoader
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class AirPollutionDataset(Dataset):
        def __init__(self, X, y=None):
            self.X = torch.tensor(X, dtype=torch.float32)

            if y is not None:
                y_tensor = torch.as_tensor(y, dtype=torch.float32)
                self.y = y_tensor.view(-1,1) if y_tensor.ndim == 1 else y_tensor
            else:
                self.y = None

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            if self.y is not None:
                return self.X[idx], self.y[idx]
            return self.X[idx]
    return AirPollutionDataset, DataLoader, nn, torch


@app.cell
def _(
    AirPollutionDataset,
    DataLoader,
    X_train_full,
    y_train_binned,
    y_train_full,
):
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_binned
    )

    train_ds = AirPollutionDataset(X_train, y_train)
    valid_ds = AirPollutionDataset(X_val, y_val)

    BATCH_SIZE = 1024
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    return BATCH_SIZE, train_ds, train_loader, valid_loader


@app.cell
def _(mo):
    mo.md(r"""
    # 3. Entrenamiento e inferencia del modelo de Red Neuronal
    """)
    return


@app.cell
def _(nn):
    class EmbeddingNet(nn.Module):
        def __init__(self, input_dim, embedding_dim=64):
            super().__init__()

            # 1. El Encoder con ResNet blocks
            self.encoder = nn.Sequential(

                nn.BatchNorm1d(input_dim),

                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.SiLU(),
                nn.Dropout(p=0.2),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Dropout(p=0.2),

                # Reducción al espacio del embedding
                nn.Linear(64, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                #nn.SiLU(),
                #nn.Dropout(p=0.3),
            )

            # 2. El Regresor con ResNet blocks
            self.regressor = nn.Sequential(

                nn.Linear(embedding_dim, 64),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Dropout(p=0.2),

                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.SiLU(),

                nn.Linear(64, 1)
            )

        def extract_feature(self, x):
            return self.encoder(x)

        def forward(self, x):
            embedding = self.encoder(x)
            prediction = self.regressor(embedding)
            return prediction

        def get_layer_groups(self):
            # Ajustamos los índices ya que la estructura ha cambiado
            return [
                self.encoder[0:5].parameters(), # Capa de proyección
                self.encoder[5:].parameters(),  # Bloque residual + reducción
                self.regressor.parameters()     # Regresor final
            ]
    return (EmbeddingNet,)


@app.cell
def _(
    EmbeddingNet,
    math,
    nn,
    np,
    time,
    torch,
    train_ds,
    train_loader,
    valid_loader,
):
    model = EmbeddingNet(input_dim=train_ds.X.shape[1]).to('cuda')

    layer_groups = model.get_layer_groups()
    n_layers = len(layer_groups)
    MAX_LR = 5e-2
    min_lr = MAX_LR / 10
    _lrs = np.logspace(
        np.log10(min_lr), 
        np.log10(MAX_LR), 
        num=n_layers
    ).tolist()

    #
    param_groups = [{'params': g, 'lr': l} for g, l in zip(layer_groups, _lrs)]
    WEIGHT_DECAY = 1e-2
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    #
    NUM_EPOCHS = 100
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=_lrs,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS
    )


    criterion = nn.MSELoss()
    _best_valid_loss = float('inf')

    # Listas para almacenar las métricas
    train_losses = []
    valid_losses = []

    # ---- Bucle de Entrenamiento------------------------------------------------
    for epoch in range(NUM_EPOCHS):
        _start_time = time.time()

        # Entrenamiento ---------------------------------------------------------
        model.train()
        _train_mse_loss = 0
        for _x_batch, _y_batch in train_loader:
            _inputs, _targets = _x_batch.to('cuda'), _y_batch.to('cuda')

            optimizer.zero_grad()
            _outputs = model(_inputs) 
            _loss = criterion(_outputs, _targets)
            _loss.backward()
            optimizer.step()
            scheduler.step()

            _train_mse_loss += _loss.item() 

        _avg_train_rmse_loss = math.sqrt(_train_mse_loss / len(train_loader))

        # Validación ------------------------------------------------------------
        model.eval()
        _valid_mse_loss = 0
        with torch.no_grad():
            for _x_val, _y_val in valid_loader:
                _x_val, _y_val = _x_val.to('cuda'), _y_val.to('cuda')
                _outputs = model(_x_val)
                _valid_mse_loss += criterion(_outputs, _y_val).item()

        _avg_valid_rmse_loss = math.sqrt(_valid_mse_loss / len(valid_loader))

        train_losses.append(_avg_train_rmse_loss)
        valid_losses.append(_avg_valid_rmse_loss)

        # Se guardan los pesos del modelo con menor pérdida en validación ----
        if _avg_valid_rmse_loss < _best_valid_loss:
            _best_valid_loss = _avg_valid_rmse_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            status = "*" 
        else:
            status = ""

        # Print de progreso ---------------------------------------------------
        _epoch_time = (time.time() - _start_time) * 1000
        print(f"Epoch {epoch+1:3d}  |  Train: {_avg_train_rmse_loss:.4f}  |  Val: {_avg_valid_rmse_loss:.4f} | {int(_epoch_time):4d}ms {status}")

    # Restaura los pesos del modelo con menor pérdida en validación -----------
    model.load_state_dict(torch.load("models/best_model.pth"))
    return (model,)


@app.cell
def _(AirPollutionDataset, BATCH_SIZE, DataLoader, X_test, model, np, torch):
    test_ds = AirPollutionDataset(X_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    _all_predictions = []

    with torch.no_grad():
        for _x_batch in test_loader:
            _x_batch = _x_batch.to('cuda')
            _prediction = model(_x_batch)

            # Guardamos ambos por separado
            _all_predictions.append(_prediction.cpu().numpy())

    # Unimos los resultados de todos los batches
    y_test_preds = np.concatenate(_all_predictions, axis=0).flatten()
    return (y_test_preds,)


@app.cell
def _(df_test, pl, y_test_preds):
    results_test = df_test.select(
        pl.col('Place_ID X Date'),
        pl.lit(y_test_preds).alias('target')
    )
    # results_test.write_csv("submission/submission_3.csv")
    results_test
    return


if __name__ == "__main__":
    app.run()
