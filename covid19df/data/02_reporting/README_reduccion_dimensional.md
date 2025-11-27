# Informe: Reducción de Dimensionalidad (PCA y t-SNE)

Este documento resume lo que ejecuta el pipeline `reduccion_dimensional` de Kedro, qué artefactos genera y cómo volver a correrlo (local y en Airflow).

## Resumen
- Fuente de datos: `df1` (CSV limpio en `data/03_intermediate/covid_19_clean_complete_CLEAN.csv`).
- Pasos principales:
  - PCA: embeddings, varianza explicada (scree), loadings y biplot (PC1 vs PC2).
  - t-SNE: visualización 2D y opcional 3D (con submuestreo para acelerar).

## Artefactos generados
- `data/03_intermediate/`
  - `pca_embeddings.csv`: coordenadas PCA por muestra (`pca_1`, `pca_2`, ...).
  - `pca_loadings.csv`: contribución de cada variable a cada componente principal.
  - `pca_explained_variance.csv`: varianza explicada por componente (para curva scree).
  - `tsne_embeddings_2d.csv`: coordenadas t‑SNE 2D (sobre una muestra si el dataset es grande).
  - `tsne_embeddings_3d.csv`: coordenadas t‑SNE 3D (si 3D está activado).
  - `X_features.parquet` (local): caché de features numéricas para corridas parciales rápidas.
- `data/05_train/`
  - `pca_biplot.png`: biplot PC1 vs PC2 con flechas (variables más relevantes).
  - `tsne_plot_2d.png`: dispersión 2D de t‑SNE.
  - `tsne_plot_3d.png`: dispersión 3D de t‑SNE (si 3D está activado).

## Cómo ejecutar (local)
```powershell
# Ejecutar todo el pipeline (PCA + t‑SNE)
kedro run --pipeline reduccion_dimensional

# Solo PCA hasta el biplot (rápido)
kedro run --pipeline reduccion_dimensional --to-nodes plot_biplot_node

# t‑SNE 2D incluyendo carga de features (parcial rápido)
kedro run --pipeline reduccion_dimensional --from-nodes load_features_node --to-nodes plot_tsne_2d_node

# Usar X_features persistido y ejecutar solo t‑SNE 2D (tras una corrida previa)
kedro run --pipeline reduccion_dimensional --from-nodes run_tsne_node --to-nodes plot_tsne_2d_node
```

## Parámetros clave (`conf/base/parameters.yml` + overrides locales en `conf/local/parameters.yml`)
- `n_pca_components`: componentes solicitados para PCA (se ajusta a `min(n_samples, n_features)`).
- `tsne_perplexity`: tamaño del vecindario efectivo (20–50 típico).
- `tsne_n_iter`: iteraciones de optimización (para pruebas usar 250–750).
- `tsne_enable_3d`: habilita/deshabilita t‑SNE 3D (costoso; por defecto local=false).
- `tsne_max_samples_2d` / `tsne_max_samples_3d`: máximo de filas usadas en t‑SNE (submuestreo para acelerar).
- `random_state`: reproducibilidad para PCA/t‑SNE.

Ejemplo de overrides locales (rápidos) en `conf/local/parameters.yml`:
```yaml
reduccion_dimensional:
  tsne_n_iter: 250
  tsne_enable_3d: false
```

## Interpretación rápida
- `pca_explained_variance.csv`: identifica cuántas PCs capturan suficiente varianza (curva scree).
- `pca_loadings.csv`: magnitud/señal de cada variable en cada PC (direcciones del biplot).
- `pca_biplot.png`: relación de muestras en PC1–PC2 y variables relevantes (flechas).
- `tsne_plot_2d.png` / `tsne_plot_3d.png`: clusters o separaciones no lineales en espacio reducido.

## Airflow
- DAG: `kedro_reduccion_dimensional_dag` (en `airflow/dags/reduccion_dimensional_dag.py`).
- Ejecuta: `kedro run --pipeline reduccion_dimensional` dentro del contenedor (misma lógica que local).

## Notas de rendimiento
- t‑SNE escala ~O(n²): por eso se submuestrea automáticamente y se permite desactivar 3D.
- Para datasets grandes, recomendar 2D con `tsne_n_iter` moderado (250–750) y ajustar `perplexity`.

## Código relevante
- Pipeline: `src/covid19df/pipelines/reduccion_dimensional/pipeline.py`
- Nodos: `src/covid19df/pipelines/reduccion_dimensional/nodes.py`
- Parámetros: `conf/base/parameters.yml` (+ `conf/local/parameters.yml` para overrides)
- Catálogo: `conf/base/catalog.yml` (+ `conf/local/catalog.yml` para `X_features`)
