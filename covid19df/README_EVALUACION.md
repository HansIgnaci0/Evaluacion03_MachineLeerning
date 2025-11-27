# Proyecto covid19DF – Evaluación Pipelines y DAGs

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Sistema MLOps modular para análisis de datos de COVID‑19 con pipelines de **regresión**, **clasificación**, **reducción de dimensionalidad (PCA + t‑SNE)** y **clustering (K‑Means, Jerárquico, GMM)**, orquestados mediante **Apache Airflow** y versionados con **DVC**.

---
## 📋 Tabla de Contenidos
1. [Características](#características)
2. [Arquitectura](#arquitectura)
3. [Requisitos](#requisitos)
4. [Instalación](#instalación)
5. [Estructura de Carpetas](#estructura-de-carpetas)
6. [Datasets y Catálogo](#datasets-y-catálogo)
7. [Parámetros Clave](#parámetros-clave)
8. [Pipelines y DAGs](#pipelines-y-dags)
9. [Ejecución Local (Kedro)](#ejecución-local-kedro)
10. [Orquestación con Airflow](#orquestación-con-airflow)
11. [Versionamiento con DVC](#versionamiento-con-dvc)
12. [Métricas y Resultados Esperados](#métricas-y-resultados-esperados)
13. [Reproducibilidad Completa](#reproducibilidad-completa)
14. [Buenas Prácticas](#buenas-prácticas)
15. [Troubleshooting](#troubleshooting)
16. [Créditos](#créditos)

---
## 🌟 Características
- Pipelines modulares con Kedro (lazy registry + aliases).
- DAGs de Airflow para ejecución automatizada:
  - `regresion_dag.py`
  - `clasificacion_dag.py`
  - `reduccion_dimensional_dag.py`
  - `kmeans_dag.py`, `jerarquico_dag.py`, `gmm_dag.py`
- Reducción de dimensionalidad: PCA (varianza explicada, loadings, biplot) + t‑SNE (2D/3D, subsampling controlado).
- Clustering múltiple con métricas de validación:
  - K‑Means (inertia + Elbow + silhouette, Davies‑Bouldin, Calinski‑Harabasz)
  - Jerárquico (Ward + dendrograma truncado + métricas)
  - Gaussian Mixture (GMM con probabilidades, BIC/AIC + métricas)
- Visualizaciones generadas y persistidas (PNG) parametrizadas vía `MatplotlibDataset`.
- Submuestreo controlado para evitar OOM en t‑SNE, Jerárquico y GMM.
- Versionamiento de outputs con DVC (stages declarados en `dvc.yaml`).
- Logging enriquecido en nodos para inspección rápida en CLI y Airflow.

---
## 🏗 Arquitectura
```
covid19df/
  airflow/dags/        # DAGs de orquestación
  conf/base/           # catalog.yml, parameters.yml (global)
  conf/local/          # overrides locales (no versionar en remoto)
  data/                # 01_raw, 03_intermediate, 05_train (artefactos)
  src/covid19df/       # código fuente (pipeline_registry, pipelines/*)
  dvc.yaml             # definición de stages reproducibles
  dvc.lock             # lock de dependencias/outs
  .dvc/                # cache y configuración DVC
```

---
## 🧩 Requisitos
- Python >= 3.10
- Kedro >= 0.19
- scikit-learn, pandas, numpy, matplotlib
- DVC >= 2.x
- Apache Airflow (opcional para DAGs)
- Docker (si se desea orquestación aislada)

---
## ⚡ Instalación
```powershell
git clone <URL_DEL_REPO>
cd covid19df
python -m venv venv_kedro
./venv_kedro/Scripts/Activate.ps1
pip install -r requirements.txt
```

Opcional inicializar DVC (ya existente):
```powershell
dvc status
```

---
## 🗂 Estructura de Carpetas
- `data/01_raw/`: datos crudos.
- `data/03_intermediate/`: features y embeddings (PCA, t‑SNE, X_features_*).
- `data/05_train/`: resultados finales (labels, métricas, plots, comparaciones).
- `conf/base/parameters.yml`: parámetros globales de todos los pipelines.
- `src/covid19df/pipelines/*`: definición de nodos y wiring.
- `airflow/dags/*.py`: definición de cada DAG.

---
## 📦 Datasets y Catálogo
`conf/base/catalog.yml` registra entradas y salidas: CSVDataset, MatplotlibDataset, Pickle y JSON. Plots usan `save_args` para controlar `dpi` y bounding box.

---
## 🔧 Parámetros Clave
Fragmentos relevantes (simplificado):
```yaml
reduccion_dimensional:
  n_pca_components: 10
  tsne_perplexity: 30
  tsne_n_iter: 1000
  tsne_max_samples_2d: 2000
  tsne_max_samples_3d: 1000
  tsne_enable_3d: true

kmeans:
  kmeans_n_clusters: 5
  kmeans_elbow_min_k: 2
  kmeans_elbow_max_k: 10

jerarquico:
  jerarquico_n_clusters: 5
  jerarquico_linkage: ward
  jerarquico_max_samples: 3000

gmm:
  gmm_n_components: 5
  gmm_covariance_type: full
  gmm_max_samples: 5000
  gmm_reg_covar: 1e-6
```

---
## 🧪 Pipelines y DAGs
| Pipeline | DAG | Propósito | Principales Outputs |
|----------|-----|-----------|---------------------|
| regresion | `regresion_dag.py` | Modelos continuos comparativos | `regression_comparison.csv/png` |
| clasificacion | `clasificacion_dag.py` | Comparación de clasificadores | `classification_comparison.csv/png` |
| reduccion_dimensional | `reduccion_dimensional_dag.py` | PCA + t‑SNE 2D/3D | `pca_*`, `tsne_embeddings_*`, plots PCA/TSNE |
| kmeans | `kmeans_dag.py` | Clustering K‑Means + Elbow + métricas | `kmeans_labels.csv`, `kmeans_metrics.csv`, `kmeans_elbow.png`, `kmeans_plot.png` |
| jerarquico | `jerarquico_dag.py` | Clustering jerárquico + dendrograma | `jerarquico_labels.csv`, `jerarquico_metrics.csv`, `jerarquico_dendrogram.png`, `jerarquico_plot.png` |
| gmm | `gmm_dag.py` | Mezcla Gaussiana probabilística | `gmm_labels.csv`, `gmm_metrics.csv`, `gmm_plot.png` |

---
## ▶️ Ejecución Local (Kedro)
Ejecutar un pipeline específico:
```powershell
./venv_kedro/Scripts/Activate.ps1
kedro run --pipeline reduccion_dimensional
kedro run --pipeline kmeans
kedro run --pipeline jerarquico
kedro run --pipeline gmm
```
Ejecutar todos:
```powershell
kedro run
```

---
## ☁ Orquestación con Airflow
1. Construir imagen (si se usa Dockerfile en `airflow/`):
```powershell
docker compose -f airflow/docker-compose.yaml up -d --build
```
2. Acceder a UI Airflow, activar DAGs: `regresion_dag`, `clasificacion_dag`, `reduccion_dimensional_dag`, `kmeans_dag`, `jerarquico_dag`, `gmm_dag`.
3. Cada DAG ejecuta `kedro run --pipeline <nombre>` dentro del contenedor.

Logs útiles: se imprimen muestras de features y métricas en cada nodo (`logger.info`).

---
## 🔁 Versionamiento con DVC
Stages declarados en `dvc.yaml` para: `kmeans_run`, `jerarquico_run`, `gmm_run` (y se pueden ampliar).
Reproducir:
```powershell
dvc repro kmeans_run
dvc repro jerarquico_run
dvc repro gmm_run
```
Subir a remote (definir previamente):
```powershell
dvc push
git add dvc.yaml dvc.lock
git commit -m "Actualiza stages DVC"
git push origin main
```

---
## 📈 Métricas y Resultados Esperados
Clustering:
- Silhouette (≈ 0 a 1): más alto mejor separación.
- Davies‑Bouldin: menor es mejor.
- Calinski‑Harabasz: mayor es mejor.
- Elbow (K‑Means): buscar el punto de inflexión de `inertia`.
- Dendrograma (Jerárquico): cortes horizontales sugieren número de clusters potencial.
- GMM: usar BIC/AIC (más bajos mejor) y probabilidades altas para confianza.

Reducción de dimensionalidad:
- PCA: revisar `pca_explained_variance.csv` y biplot para componentes relevantes.
- t‑SNE: embeddings 2D/3D para estructura local; subsampling reduce tiempos.

Regresión / Clasificación: archivos `*_comparison.csv/png` con ranking por métrica (RMSE, accuracy, etc.).

Interpretación rápida:
- Si un método produce un cluster masivo y muchos mínimos, considerar normalizar previo al fit (ya se estandariza para métricas/plots) o ajustar hiperparámetros.
- Si silhouette < 0.1: clusters poco útiles; revisar k o cambiar algoritmo.

---
## ♻️ Reproducibilidad Completa
1. Clonar repositorio y crear entorno.
2. `dvc pull` (si se comparte remote) para obtener artefactos.
3. `kedro run --pipeline <pipeline>` o `dvc repro <stage>` para regenerar outputs.
4. Comparar cambios con `dvc diff` antes de hacer push.

---
## ✅ Buenas Prácticas
- Mantener datos crudos inmutables en `data/01_raw/`.
- Usar parámetros en `conf/local/` para ejecuciones rápidas (submuestreo) sin modificar base.
- Versionar sólo artefactos finales vía DVC stages (no subir binarios grandes a Git).
- Revisar logs de nodos para validaciones rápidas sin abrir archivos.
- Añadir nuevas métricas en nodos separados para no romper reproducibilidad.

---
## 🛠 Troubleshooting
| Problema | Causa | Solución |
|----------|-------|----------|
| `ValueError` GMM covarianza | Singularidad | Incrementar `gmm_reg_covar` o reducir `gmm_n_components` |
| t‑SNE muy lento | Demasiadas filas | Bajar `tsne_max_samples_*` o `tsne_n_iter` |
| Jerárquico OOM | Complejidad O(n^2) | Ajustar `jerarquico_max_samples` |
| Métricas NaN | 1 solo cluster | Ajustar k / parámetros para generar más de un cluster |
| DVC conflicto outs | Doble tracking | Ejecutar `dvc remove <.dvc>` y usar stage en `dvc.yaml` |

---
## 👤 Créditos
Autor: Hans Ignacio Mancilla Sandoval  
Contacto: ha.mancilla@duocuc.cl  
Asignatura: Machine Learning  
Profesor: Giocrisrai Godoy  

---
## 📄 Licencia
Proyecto académico; uso educativo y demostrativo.

---
## 🔍 Referencias
- Kedro Docs: https://docs.kedro.org
- DVC Docs: https://dvc.org/doc
- scikit-learn Cluster Metrics: https://scikit-learn.org/stable/modules/clustering.html

---
## 🚀 Comandos Rápidos
```powershell
./venv_kedro/Scripts/Activate.ps1
kedro run --pipeline reduccion_dimensional
kedro run --pipeline kmeans
dvc repro kmeans_run
```

---
## 🧪 Extensiones Futuras
- Añadir UMAP como alternativa a t‑SNE.
- Elipses de confianza en plot GMM.
- Export de métricas a JSON para `dvc metrics show`.
