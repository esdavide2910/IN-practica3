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
    return StratifiedKFold, math, np, optuna, pl, root_mean_squared_error, time


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
def _(df_X_test_1, df_X_train_1, numerical_features_cols):
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer, SimpleImputer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.compose import ColumnTransformer

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
    df_X_test_2 = _ct.transform(df_X_test_1)
    return df_X_test_2, df_X_train_2


@app.cell
def _(df_X_train_2):
    df_X_train_2
    return


@app.cell
def _(df_X_test_2):
    df_X_test_2
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
    # 3. Entrenamiento e inferencia del modelo XGBoost con embeddings
    """)
    return


@app.cell
def _(nn):
    class EmbeddingNet(nn.Module):
        def __init__(self, input_dim, embedding_dim=48):
            super().__init__()

            # 1. El Encoder con ResNet blocks
            self.encoder = nn.Sequential(

                nn.BatchNorm1d(input_dim),

                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.SiLU(),
                nn.Dropout(p=0.25),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Dropout(p=0.25),

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
                nn.Dropout(p=0.25),

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
    MAX_LR = 1e-2
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
def _(model, np, torch, train_loader, valid_loader):
    def get_embeddings(model, loader, device='cuda', has_targets=True):
        model.eval()
        embeddings = []
        targets = []

        with torch.no_grad():
            for batch in loader:
                # Manejar si el loader devuelve (X, y) o solo X
                if has_targets:
                    x_batch, y_batch = batch
                    targets.append(y_batch.numpy())
                else:
                    x_batch = batch

                x_batch = x_batch.to(device)
                emb = model.extract_feature(x_batch)
                embeddings.append(emb.cpu().numpy())

        X_emb = np.vstack(embeddings)
        if has_targets:
            y_emb = np.vstack(targets).flatten()
            return X_emb, y_emb
        return X_emb

    X_train_emb, y_train_emb = get_embeddings(model, train_loader, has_targets=True)
    X_val_emb, y_val_emb = get_embeddings(model, valid_loader, has_targets=True)
    return X_train_emb, X_val_emb, get_embeddings, y_train_emb, y_val_emb


@app.cell
def _(X_train_emb, X_val_emb, np, y_train_emb, y_val_emb):
    X_train_full_emb = np.vstack([X_train_emb, X_val_emb])
    y_train_full_emb = np.hstack([y_train_emb, y_val_emb])
    return X_train_full_emb, y_train_full_emb


@app.cell
def _(
    StratifiedKFold,
    X_train_full_emb,
    np,
    optuna,
    root_mean_squared_error,
    y_train_binned,
    y_train_full_emb,
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
        "n_jobs": 1,
        "early_stopping": 30,
    }

    def objective_XGB(trial, X, y, y_binned):

        tuned_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.5, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 1000, 1100),
            "subsample": trial.suggest_float("subsample", 0.7, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.8),
            "max_depth": trial.suggest_int("max_depth", 7, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 12),
            "gamma": trial.suggest_float("gamma", 1, 3),
        }

        param = {**FIXED_PARAMS_XGB, **tuned_params}

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
            X_t, X_v = X[train_idx], X[val_idx]
            y_t, y_v = y[train_idx], y[val_idx]

            model = xgb.XGBRegressor(**param)
            model.fit(
                X_t, y_t,
                eval_set=[(X_v, y_v)],
                verbose=False,
            )

            preds = model.predict(X_v)
            rmse = root_mean_squared_error(y_v, preds)
            rmse_scores.append(rmse)

            trial.report(rmse, fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(rmse_scores)


    study_XGB = optuna.create_study(
        direction='minimize', 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )

    study_XGB.optimize(
        lambda trial: objective_XGB(trial, X_train_full_emb, y_train_full_emb, y_train_binned), 
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
def _(
    AirPollutionDataset,
    BATCH_SIZE,
    DataLoader,
    X_test,
    X_train_full_emb,
    best_params_XGB,
    get_embeddings,
    model,
    xgb,
    y_train_full_emb,
):
    final_model = xgb.XGBRegressor(**best_params_XGB)
    final_model.fit(X_train_full_emb, y_train_full_emb)

    # Extraer embeddings de Test
    test_ds = AirPollutionDataset(X_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    X_test_emb = get_embeddings(model, test_loader, has_targets=False)

    # Predicción final con XGBoost
    test_preds = final_model.predict(X_test_emb)
    return (test_preds,)


@app.cell
def _(df_test, pl, test_preds):
    results_test = df_test.select(
        pl.col('Place_ID X Date'),
        pl.lit(test_preds).alias('target')
    )
    results_test.write_csv("submission/submission_15.csv")
    results_test
    return


if __name__ == "__main__":
    app.run()
