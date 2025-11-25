import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def load_and_prepare_data(covid_csv: str, usa_csv: str, n_rows: int = 20000):
    df_covid = pd.read_csv(covid_csv).head(n_rows)
    df_usa = pd.read_csv(usa_csv).head(n_rows)

    covid_cols = [c for c in df_covid.columns if df_covid[c].dtype != 'O']
    usa_cols = [c for c in df_usa.columns if df_usa[c].dtype != 'O']

    df_covid_sel = df_covid[covid_cols].fillna(0)
    df_usa_sel = df_usa[usa_cols].fillna(0)

    df_covid_sel = df_covid_sel.add_suffix("_covid")
    df_usa_sel = df_usa_sel.add_suffix("_usa")

    df_combined = pd.concat([df_covid_sel, df_usa_sel], axis=1)

    if "Deaths_covid" not in df_combined.columns:
        target = df_combined.columns[0]
    else:
        target = "Deaths_covid"

    X = df_combined.drop(columns=[target])
    y = df_combined[target]

    print(f"✅ Dataset combinado con {len(df_combined)} filas y {len(df_combined.columns)} columnas.")
    return X, y


def _metrics_heatmap_figure(df: pd.DataFrame, title: str):
    """Crea una figura tipo heatmap (sin seaborn) a partir de un DataFrame de métricas."""
    met_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(1.8 + 1.2 * met_df.shape[1], 1.5 + 0.6 * met_df.shape[0]))
    im = ax.imshow(met_df.values, cmap="viridis")
    ax.set_xticks(range(met_df.shape[1]))
    ax.set_xticklabels(met_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(met_df.shape[0]))
    ax.set_yticklabels(df["model"].tolist() if "model" in df.columns else df.index.tolist())
    ax.set_title(title)
    for i in range(met_df.shape[0]):
        for j in range(met_df.shape[1]):
            ax.text(j, i, f"{met_df.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
    plt.tight_layout()
    return fig


def train_models(X, y):
    """Compatibilidad: mantiene el entrenamiento simple e imprime métricas básicas."""
    comparison_df, fig = train_models_gridsearch(X, y, save_plots=False)
    print("\n📊 Resumen (test set, GridSearchCV):")
    for _, row in comparison_df.iterrows():
        print(
            f"   • {row['model']:<20} → R2: {row['test_r2']:.4f} | MAE: {row['test_mae']:.4f} | RMSE: {row['test_rmse']:.4f}"
        )
    return comparison_df


def train_models_gridsearch(X, y, save_plots: bool = True):
    """
    Entrena 5 modelos de regresión con GridSearchCV (KFold k>=5),
    calcula métricas y devuelve una tabla comparativa y una figura heatmap.

    Outputs:
    - comparison_df: DataFrame con columnas [model, best_params, cv_r2_mean, test_r2, test_mae, test_rmse]
    - fig: matplotlib.figure.Figure (heatmap de métricas)
    """
    if len(X) < 10:
        print(f"⚠️ Dataset demasiado pequeño ({len(X)} muestras). No se puede dividir en train/test.")
        empty_df = pd.DataFrame(columns=["model", "best_params", "cv_r2_mean", "test_r2", "test_mae", "test_rmse"]) 
        fig = plt.figure()
        return empty_df, fig

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    models_and_grids = {
        "LinearRegression": (
            LinearRegression(),
            {"fit_intercept": [True, False], "positive": [False]}
        ),
        "DecisionTree": (
            DecisionTreeRegressor(random_state=42),
            {"max_depth": [None, 5, 10], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]}
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [None, 10], "min_samples_split": [2, 5]}
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}
        ),
        "KNeighbors": (
            KNeighborsRegressor(),
            {"n_neighbors": [3, 5, 11], "weights": ["uniform", "distance"]}
        ),
    }

    rows = []
    print("\n� GridSearchCV (k=5) para modelos de regresión:")
    for name, (model, grid) in models_and_grids.items():
        gs = GridSearchCV(
            estimator=model,
            param_grid=grid,
            scoring="r2",
            cv=kfold,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        gs.fit(X_train_scaled, y_train)
        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        rows.append(
            {
                "model": name,
                "best_params": str(gs.best_params_),
                "cv_r2_mean": float(gs.best_score_),
                "test_r2": float(r2),
                "test_mae": float(mae),
                "test_rmse": rmse,
            }
        )
        print(f"   • {name:<16} → CV R2: {gs.best_score_:.4f} | test R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    comparison_df = pd.DataFrame(rows).sort_values(by=["test_r2", "cv_r2_mean"], ascending=[False, False]).reset_index(drop=True)
    fig = _metrics_heatmap_figure(
        comparison_df[["model", "cv_r2_mean", "test_r2", "test_mae", "test_rmse"]].copy(),
        title="Regresión: comparación de métricas"
    ) if save_plots else plt.figure()

    return comparison_df, fig
