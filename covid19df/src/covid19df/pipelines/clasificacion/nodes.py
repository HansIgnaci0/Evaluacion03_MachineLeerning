import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def load_and_prepare_data_classif(covid_csv: str, usa_csv: str, n_rows: int = 20000):
    """
    Carga y combina features numéricas de dos CSVs (covid y usa) igual que en regresión,
    y construye un objetivo binario para clasificación basado en Deaths_covid.

    - X: todas las columnas numéricas de ambos CSVs, con sufijos "_covid" y "_usa".
    - y: 1 si Deaths_covid > 0; 0 en caso contrario. Si no existe Deaths_covid, se
         toma la primera columna numérica de covid como target y se binariza por > 0.
    """
    # Cargar CSVs
    df_covid = pd.read_csv(covid_csv).head(n_rows)
    df_usa = pd.read_csv(usa_csv).head(n_rows)

    # Solo columnas numéricas
    covid_cols = [c for c in df_covid.columns if df_covid[c].dtype != 'O']
    usa_cols = [c for c in df_usa.columns if df_usa[c].dtype != 'O']

    df_covid_sel = df_covid[covid_cols].fillna(0).add_suffix("_covid")
    df_usa_sel = df_usa[usa_cols].fillna(0).add_suffix("_usa")

    # Combinar horizontalmente
    df_combined = pd.concat([df_covid_sel, df_usa_sel], axis=1)

    # Target binario basado en Deaths_covid
    target_col = "Deaths_covid" if "Deaths_covid" in df_combined.columns else df_covid_sel.columns[0]
    y_raw = df_combined[target_col]
    y = (y_raw > 0).astype(int)
    X = df_combined.drop(columns=[target_col])

    print(f"✅ Dataset combinado (clasificación) con {len(df_combined)} filas y {len(df_combined.columns)} columnas.")
    pos_rate = y.mean()
    print(f"ℹ️  Proporción de clase positiva (y=1): {pos_rate:.3f}")
    return X, y


def _metrics_heatmap_figure(df: pd.DataFrame, title: str):
    # Asegurar solo columnas numéricas
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


def train_classification_models(X, y):
    """
    Entrena 5 modelos de clasificación y muestra métricas clave.
    Métricas: accuracy, F1 (binario) y ROC-AUC (si el modelo expone predict_proba).
    """
    # Validar tamaño del dataset
    if len(X) < 10:
        print(f"⚠️ Dataset demasiado pequeño ({len(X)} muestras). No se puede dividir en train/test.")
        return {}

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Escalado (beneficia a LR/KNN). Árboles no lo requieren pero no afecta negativamente.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=2000, n_jobs=None),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'KNeighbors': KNeighborsClassifier()
    }

    results = {}

    print("\n📊 Resultados de modelos de clasificación:")
    for name, model in models.items():
        # Usamos datos escalados para todos por simplicidad
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC-AUC si hay probabilidades
        try:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = float('nan')

        results[name] = {'accuracy': acc, 'f1': f1, 'roc_auc': auc}
        auc_txt = f" | AUC: {auc:.4f}" if auc == auc else ""
        print(f"   • {name:<18} → Acc: {acc:.4f} | F1: {f1:.4f}{auc_txt}")

    return results


def train_classification_models_gridsearch(X, y, save_plots: bool = True):
    """
    Entrena 5 clasificadores con GridSearchCV (StratifiedKFold k>=5),
    devuelve una tabla comparativa y figura heatmap.

    Outputs:
    - comparison_df: [model, best_params, cv_f1_mean, test_accuracy, test_f1, test_roc_auc]
    - fig: matplotlib.figure.Figure
    """
    if len(X) < 10:
        print(f"⚠️ Dataset demasiado pequeño ({len(X)} muestras). No se puede dividir en train/test.")
        empty_df = pd.DataFrame(columns=["model", "best_params", "cv_f1_mean", "test_accuracy", "test_f1", "test_roc_auc"]) 
        fig = plt.figure()
        return empty_df, fig

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models_and_grids = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, n_jobs=None, solver="lbfgs"),
            {"C": [0.1, 1.0, 10.0]}
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [None, 5, 10], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [None, 10], "min_samples_split": [2, 5]}
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}
        ),
        "KNeighbors": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 11], "weights": ["uniform", "distance"]}
        ),
    }

    rows = []
    print("\n🔎 GridSearchCV (k=5) para modelos de clasificación:")
    for name, (model, grid) in models_and_grids.items():
        gs = GridSearchCV(
            estimator=model,
            param_grid=grid,
            scoring="f1",
            cv=skf,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        gs.fit(X_train_scaled, y_train)
        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        # ROC-AUC si hay probabilidades
        try:
            y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = float('nan')

        rows.append(
            {
                "model": name,
                "best_params": str(gs.best_params_),
                "cv_f1_mean": float(gs.best_score_),
                "test_accuracy": float(acc),
                "test_f1": float(f1),
                "test_roc_auc": float(auc) if auc == auc else np.nan,
            }
        )
        auc_txt = f" | AUC: {auc:.4f}" if auc == auc else ""
        print(f"   • {name:<16} → CV F1: {gs.best_score_:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}{auc_txt}")

    comparison_df = pd.DataFrame(rows).sort_values(by=["test_f1", "cv_f1_mean"], ascending=[False, False]).reset_index(drop=True)
    fig = _metrics_heatmap_figure(
        comparison_df[["model", "cv_f1_mean", "test_accuracy", "test_f1", "test_roc_auc"]].copy(),
        title="Clasificación: comparación de métricas"
    ) if save_plots else plt.figure()

    return comparison_df, fig
