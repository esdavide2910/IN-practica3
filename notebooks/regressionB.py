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
    import time
    import math

    #
    import polars as pl
    import numpy as np

    #
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go

    #
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import KFold
    import lightgbm as lgb
    from sklearn.metrics import root_mean_squared_error

    #
    import optuna

    #
    import warnings
    warnings.filterwarnings("ignore")
    return (
        DataLoader,
        Dataset,
        KFold,
        go,
        lgb,
        math,
        nn,
        np,
        optuna,
        pl,
        root_mean_squared_error,
        time,
        torch,
        train_test_split,
    )


@app.cell
def _(torch):
    # Comprobar si CUDA está disponible
    print(f"CUDA disponible: {torch.cuda.is_available()}")

    # Ver el número de GPUs disponibles
    print(f"Número de GPUs: {torch.cuda.device_count()}")

    # Ver el nombre de la GPU actual (si hay alguna)
    if torch.cuda.is_available():
        print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")
        print(f"ID de la GPU actual: {torch.cuda.current_device()}")
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
    ## 2.2. División de datos en características y valores objetivos
    """)
    return


@app.cell
def _(df_test, df_train):
    #
    df_X_train = df_train.drop(['Place_ID','Date', 'Place_ID X Date', 
                             'target','target_min','target_max','target_variance','target_count'])
    df_y_train = df_train['target']
    #
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
    ## 2.3. Preparación del input para el modelo
    """)
    return


@app.cell
def _(np):
    class NullMaskTransformer:
        def __init__(self):
            self.indices_with_null = None

        def fit(self, X):
            # Identifica qué columnas tienen nulos solo en el set de FIT (train)
            nulls_per_col = np.isnan(X).any(axis=0)
            self.indices_with_null = np.where(nulls_per_col)[0]
            return self

        def transform(self, X):
            # Si no hay columnas con nulos, devuelve X con ceros
            if len(self.indices_with_null) == 0:
                return np.nan_to_num(X, nan=0.0)

            # Crea la máscara usando los índices guardados en el fit
            X_nulls = X[:, self.indices_with_null]
            # 1.0 si existe el dato, 0.0 si es nulo
            nulls_mask = (~np.isnan(X_nulls)).astype(np.float32)

            # Limpia originales
            X_clean = np.nan_to_num(X, nan=0.0)

            return np.concatenate([X_clean, nulls_mask], axis=1)

        def fit_transform(self, X):
            return self.fit(X).transform(X)
    return (NullMaskTransformer,)


@app.cell
def _(Dataset, torch):
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
    return (AirPollutionDataset,)


@app.cell
def _(
    AirPollutionDataset,
    DataLoader,
    NullMaskTransformer,
    df_X_test_norm,
    df_X_train_norm,
    df_y_train,
    pl,
    train_test_split,
):
    _null_mask_transformer = NullMaskTransformer()
    X_train = _null_mask_transformer.fit_transform(df_X_train_norm)
    X_test = _null_mask_transformer.transform(df_X_test_norm)

    y_labels = pl.Series(df_y_train).qcut(10)
    y_train = df_y_train.to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_labels
    )

    train_ds = AirPollutionDataset(X_train, y_train)
    valid_ds = AirPollutionDataset(X_val, y_val)

    BATCH_SIZE = 1024
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    return BATCH_SIZE, X_test, train_ds, train_loader, valid_loader


@app.cell
def _(mo):
    mo.md(r"""
    # 3. Arquitectura neuronal
    """)
    return


@app.cell
def _(nn):
    # class ResidualBlock(nn.Module):
    #     def __init__(self, in_features, out_features, dropout_rate=0.2):
    #         super().__init__()

    #         self.bn1 = nn.BatchNorm1d(in_features)
    #         self.linear1 = nn.Linear(in_features, out_features)

    #         self.bn2 = nn.BatchNorm1d(out_features)
    #         self.linear2 = nn.Linear(out_features, out_features)

    #         self.dropout = nn.Dropout(dropout_rate)
    #         self.activation = nn.SiLU()

    #         self.skip = (
    #             nn.Linear(in_features, out_features)
    #             if in_features != out_features
    #             else nn.Identity()
    #         )

    #         # Inicialización residual estable
    #         nn.init.zeros_(self.linear2.weight)
    #         nn.init.zeros_(self.linear2.bias)

    #     def forward(self, x):
    #         identity = self.skip(x)

    #         out = self.bn1(x)
    #         out = self.activation(out)
    #         out = self.linear1(out)

    #         out = self.bn2(out)
    #         out = self.activation(out)
    #         out = self.linear2(out)

    #         out = self.dropout(out)

    #         return out + identity

    # class EmbeddingNet(nn.Module):
    #     def __init__(self, input_dim, embedding_dim=64):
    #         super().__init__()

    #         self.input_projection = nn.Sequential(
    #             nn.Linear(input_dim, 256),
    #             nn.BatchNorm1d(256),
    #             nn.SiLU(),
    #             nn.Dropout(0.3)
    #         )

    #         self.res_blocks = nn.ModuleList([
    #             ResidualBlock(256, 128, dropout_rate=0.4),
    #             ResidualBlock(128, embedding_dim, dropout_rate=0.4),
    #         ])

    #         self.embedding_norm = nn.BatchNorm1d(embedding_dim)
    #         self.embedding_dropout = nn.Dropout(0.2)

    #         self.regressor = nn.Sequential(
    #             nn.Linear(embedding_dim, 64),
    #             nn.BatchNorm1d(64),
    #             nn.SiLU(),
    #             nn.Dropout(0.2),
    #             nn.Linear(64, 1)
    #         )

    #     def extract_feature(self, x):
    #         x = self.input_projection(x)
    #         for block in self.res_blocks:
    #             x = block(x)
    #         x = self.embedding_norm(x)
    #         x = self.embedding_dropout(x)
    #         return x

    #     def forward(self, x):
    #         return self.regressor(self.extract_feature(x))

    #     def get_layer_groups(self):
    #         """
    #         Grupos de capas para discriminative learning rates
    #         (de más cercano al input a más cercano al output)
    #         """

    #         return [
    #             # Grupo 1: Proyección inicial (muy estable)
    #             self.input_projection.parameters(),

    #             # Grupo 2: Encoder residual (feature extractor)
    #             self.res_blocks.parameters(),

    #             # Grupo 3: Normalización del embedding
    #             self.embedding_norm.parameters(),

    #             # Grupo 4: Regressor hidden layers
    #             list(self.regressor[:-1].parameters()),

    #             # Grupo 5: Capa final (salida)
    #             self.regressor[-1].parameters()
    #         ]


    class EmbeddingNet(nn.Module):
        def __init__(self, input_dim, embedding_dim=64):
            super().__init__()

            # 1. El Encoder con ResNet blocks
            self.encoder = nn.Sequential(

                nn.BatchNorm1d(input_dim),

                # nn.Linear(input_dim, 256),
                # nn.BatchNorm1d(256),
                # nn.SiLU(),
                # nn.Dropout(p=0.35),

                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Dropout(p=0.2),

                # Reducción al espacio del embedding
                nn.Linear(64, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.SiLU(),
                nn.Dropout(p=0.4),
            )

            # 2. El Regresor con ResNet blocks
            self.regressor = nn.Sequential(

                nn.Linear(embedding_dim, 64),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Dropout(p=0.4),

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
def _(mo):
    mo.md(r"""
    # 4. Entrenamiento de la red
    """)
    return


@app.cell
def _(np, torch):
    def apply_mixup(x, y, alpha=0.3):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        # Mezcla lineal de dos filas y sus respectivos targets
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_x, mixed_y
    return (apply_mixup,)


@app.cell
def _(
    EmbeddingNet,
    apply_mixup,
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
    # optimizer = torch.optim.RAdam(param_groups, weight_decay=WEIGHT_DECAY)

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
            _x_batch, _y_batch = _x_batch.to('cuda'), _y_batch.to('cuda')

            # Aplicar Mixup
            _inputs, _targets = apply_mixup(_x_batch, _y_batch)

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
            torch.save(model.state_dict(), "models/best_model_B1.pth")
            status = "*" 
        else:
            status = ""

        # Print de progreso ---------------------------------------------------
        _epoch_time = (time.time() - _start_time) * 1000
        print(f"Epoch {epoch+1:3d} | Train: {_avg_train_rmse_loss:.4f} | Val: {_avg_valid_rmse_loss:.4f} | {int(_epoch_time):4d}ms {status}")

    # Restaura los pesos del modelo con menor pérdida en validación -----------
    model.load_state_dict(torch.load("models/best_model_B1.pth"))
    return model, train_losses, valid_losses


@app.cell
def _(go, train_losses, valid_losses):
    _fig = go.Figure()

    # Curva de Entrenamiento
    _fig.add_trace(go.Scatter(
        x=list(range(1, len(train_losses) + 1)),
        y=train_losses,
        mode='lines+markers',
        name='Train RMSE',
        line=dict(color='royalblue', width=2)
    ))

    # Curva de Validación
    _fig.add_trace(go.Scatter(
        x=list(range(1, len(valid_losses) + 1)),
        y=valid_losses,
        mode='lines+markers',
        name='Val RMSE',
        line=dict(color='firebrick', width=2)
    ))

    # Estética del gráfico
    _fig.update_layout(
        title='Curvas de Entrenamiento y Validación (RMSE)',
        xaxis_title='Época',
        yaxis_title='Loss (RMSE)',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode='x unified'
    )

    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 5. Inferencia del modelo final
    """)
    return


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
    return test_loader, y_test_preds


@app.cell
def _(y_test_preds):
    y_test_preds
    return


@app.cell
def _(df_test, pl, y_test_preds):
    results_test = df_test.select(
        pl.col('Place_ID X Date'),
        pl.lit(y_test_preds).alias('target')
    )
    results_test.write_csv("submission/submission_3.csv")
    results_test
    return (results_test,)


@app.cell
def _(mo):
    mo.md(r"""
    # 6. Modelo con LightGBM
    """)
    return


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

    # --- 2. Extracción de datos procesados ---
    X_train_emb, y_train_emb = get_embeddings(model, train_loader, has_targets=True)
    X_val_emb, y_val_emb = get_embeddings(model, valid_loader, has_targets=True)

    # Unimos para el Cross-Validation de Optuna
    X_full_emb = np.vstack([X_train_emb, X_val_emb])
    y_full_emb = np.concatenate([y_train_emb, y_val_emb])
    return X_full_emb, get_embeddings, y_full_emb


@app.cell
def _(KFold, X_full_emb, lgb, np, optuna, root_mean_squared_error, y_full_emb):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial, X, y):
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_leaves': trial.suggest_int('num_leaves', 32, 256),
            'max_depth': trial.suggest_int('max_depth', 5, 15), 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []

        for train_idx, val_idx in kf.split(X):
            X_t, X_v = X[train_idx], X[val_idx]
            y_t, y_v = y[train_idx], y[val_idx]

            lgb_reg = lgb.LGBMRegressor(**param)
            lgb_reg.fit(
                X_t, y_t,
                eval_set=[(X_v, y_v)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10, verbose=0),
                    lgb.log_evaluation(0)
                ]
            )

            preds = lgb_reg.predict(X_v)
            rmse = root_mean_squared_error(y_v, preds)
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    # Ejecutar el estudio
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_full_emb, y_full_emb), 
        n_trials=20,
        show_progress_bar=True
    )
    return (study,)


@app.cell
def _(study):
    print("Mejores hiperparámetros:", study.best_params)
    print("Mejor RMSE:", study.best_value)
    return


@app.cell
def _(X_full_emb, get_embeddings, lgb, model, study, test_loader, y_full_emb):
    best_params = study.best_params

    final_model = lgb.LGBMRegressor(**best_params, n_estimators=1000, random_state=42)
    final_model.fit(X_full_emb, y_full_emb)

    # b. Extraer embeddings de Test
    X_test_emb = get_embeddings(model, test_loader, has_targets=False)

    # c. Predicción final con LightGBM
    test_preds = final_model.predict(X_test_emb)
    return (test_preds,)


@app.cell
def _(df_test, pl, results_test, test_preds):
    results_test2 = df_test.select(
        pl.col('Place_ID X Date'),
        pl.lit(test_preds).alias('target')
    )
    results_test.write_csv("submission/submission_4.csv")
    results_test2
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
