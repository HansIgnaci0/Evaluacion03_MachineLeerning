#!/usr/bin/env python3
"""
Ad-hoc runner to execute clustering + reduction node functions
without requiring Kedro to be installed or `pipeline_registry` to be imported.

Run from the project (or any sub) folder. It will try to auto-detect project root.
"""
from pathlib import Path
import sys
import os
import traceback


def find_project_root(start=Path.cwd()):
    cur = Path(start).resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "src").is_dir() or (parent / "pyproject.toml").is_file() or (parent / "conf").is_dir():
            return parent
    return Path.cwd()


def main():
    ROOT = find_project_root()
    print("Detected project root:", ROOT)
    # ensure package imports work
    sys.path.insert(0, str(ROOT / "src"))

    # paths
    input_csv = str(ROOT / "data" / "03_intermediate" / "covid_19_clean_complete_CLEAN.csv")
    out_dir = str(ROOT / "data" / "05_train")
    os.makedirs(out_dir, exist_ok=True)

    # outputs
    elbow_csv = os.path.join(out_dir, "elbow.csv")
    elbow_plot = os.path.join(out_dir, "elbow.png")
    kmeans_csv = os.path.join(out_dir, "kmeans_clusters.csv")
    kmeans_metrics = os.path.join(out_dir, "kmeans_metrics.csv")

    try:
        # Import only the node functions we need (they should not import Kedro)
        from covid19df.pipelines.unsupervised.clustering.nodes import (
            load_latest_by_country,
            preprocess_for_clustering,
            elbow_method,
            run_kmeans,
            run_dbscan,
            run_hierarchical,
            plot_dendrogram,
            compute_clustering_metrics,
            save_cluster_results,
            save_metrics_csv,
        )

        from covid19df.pipelines.unsupervised.reduction.nodes import run_pca, save_embeddings
        from covid19df.pipelines.unsupervised.reduction.nodes import run_tsne
    except Exception:
        print("Failed importing node functions. Traceback:")
        traceback.print_exc()
        print("If imports fail, ensure you're running this with the project's venv where requirements are installed.")
        return

    print("Input exists:", os.path.exists(input_csv))

    try:
        raw = load_latest_by_country(input_csv)
        X_scaled_df, raw_df = preprocess_for_clustering(raw)
        print("Loaded shapes -> raw:", raw_df.shape, "features:", X_scaled_df.shape)

        print("Running elbow method...")
        elbow_res = elbow_method(X_scaled_df, k_min=2, k_max=8, out_csv=elbow_csv, out_plot=elbow_plot)
        print("Saved elbow to:", elbow_csv)

        k = 4
        print(f"Running KMeans (k={k})...")
        k_labels, k_model = run_kmeans(X_scaled_df, n_clusters=k)
        save_cluster_results(raw_df, k_labels, kmeans_csv)
        k_metrics = compute_clustering_metrics(X_scaled_df, k_labels)
        save_metrics_csv(k_metrics, kmeans_metrics)
        print("KMeans outputs:", kmeans_csv, kmeans_metrics)

        # DBSCAN
        dbscan_csv = os.path.join(out_dir, "dbscan_clusters.csv")
        dbscan_metrics = os.path.join(out_dir, "dbscan_metrics.csv")
        print("Running DBSCAN (eps=0.5, min_samples=5)...")
        try:
            db_labels, db_model = run_dbscan(X_scaled_df, eps=0.5, min_samples=5)
            save_cluster_results(raw_df, db_labels, dbscan_csv)
            db_metrics = compute_clustering_metrics(X_scaled_df, db_labels)
            save_metrics_csv(db_metrics, dbscan_metrics)
            print("DBSCAN outputs:", dbscan_csv, dbscan_metrics)
        except Exception:
            print("DBSCAN failed:")
            traceback.print_exc()

        # Hierarchical (Agglomerative) + dendrogram
        hier_csv = os.path.join(out_dir, "hier_clusters.csv")
        dendrogram_plot = os.path.join(out_dir, "dendrogram.png")
        hier_metrics = os.path.join(out_dir, "hier_metrics.csv")
        print("Running Agglomerative (n_clusters=4)...")
        try:
            hier_labels, hier_model = run_hierarchical(X_scaled_df, n_clusters=k, linkage="ward")
            save_cluster_results(raw_df, hier_labels, hier_csv)
            h_metrics = compute_clustering_metrics(X_scaled_df, hier_labels)
            save_metrics_csv(h_metrics, hier_metrics)
            print("Hierarchical outputs:", hier_csv, hier_metrics)
            print("Saving dendrogram to:", dendrogram_plot)
            pdend = plot_dendrogram(X_scaled_df, method="ward", out_plot=dendrogram_plot)
            if pdend:
                print("Dendrogram saved:", pdend)
        except Exception:
            print("Hierarchical clustering failed:")
            traceback.print_exc()

        print("Running PCA (2 components)...")
        pca_res = run_pca(X_scaled_df, n_components=2)
        import numpy as np
        pca_emb_path = os.path.join(out_dir, "pca_embeddings.csv")
        save_embeddings(raw_df, np.array(pca_res["embeddings"]), pca_emb_path)
        print("PCA embeddings saved:", pca_emb_path)

        # try t-SNE
        tsne_emb_path = os.path.join(out_dir, "tsne_embeddings.csv")
        tsne_plot = os.path.join(out_dir, "tsne_scatter.png")
        try:
            print("Running t-SNE (2 components)...")
            tsne_res = run_tsne(X_scaled_df, n_components=2, perplexity=30.0)
            save_embeddings(raw_df, np.array(tsne_res["embeddings"]), tsne_emb_path)
            print("t-SNE embeddings saved:", tsne_emb_path)
        except Exception:
            print("t-SNE failed:")
            traceback.print_exc()

        # Create scatter plots colored by cluster for KMeans, DBSCAN, Hierarchical
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np

            def _save_scatter(embeddings, labels, out_file, title):
                plt.figure(figsize=(6, 5))
                df_plot = None
                try:
                    import pandas as pd
                    df_plot = pd.DataFrame(embeddings, columns=["dim1", "dim2"])
                    df_plot["cluster"] = labels
                except Exception:
                    df_plot = None

                if df_plot is not None:
                    sns.scatterplot(data=df_plot, x="dim1", y="dim2", hue="cluster", palette="tab10", legend="brief")
                else:
                    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="tab10")
                plt.title(title)
                plt.tight_layout()
                plt.savefig(out_file)
                plt.close()

            # KMeans scatter (PCA space)
            try:
                _save_scatter(np.array(pca_res["embeddings"]), k_labels, os.path.join(out_dir, "pca_kmeans_scatter.png"), f"PCA scatter colored by KMeans (k={k})")
            except Exception:
                pass

            # t-SNE scatter for KMeans cluster coloring if available
            try:
                import numpy as _np
                if os.path.exists(tsne_emb_path):
                    emb = _np.loadtxt(tsne_emb_path, delimiter=",", skiprows=1, usecols=(1,2)) if False else None
                    # better: read with pandas
                    import pandas as _pd
                    embdf = _pd.read_csv(tsne_emb_path)
                    emb = embdf[[col for col in embdf.columns if col.startswith('dim_')]].values
                    _save_scatter(emb, k_labels, os.path.join(out_dir, "tsne_kmeans_scatter.png"), f"t-SNE colored by KMeans (k={k})")
            except Exception:
                pass

            # DBSCAN scatter if available
            try:
                if os.path.exists(tsne_emb_path):
                    embdf = __import__('pandas').read_csv(tsne_emb_path)
                    emb = embdf[[col for col in embdf.columns if col.startswith('dim_')]].values
                    _save_scatter(emb, db_labels, os.path.join(out_dir, "tsne_dbscan_scatter.png"), "t-SNE colored by DBSCAN")
            except Exception:
                pass

            # Hierarchical scatter
            try:
                if os.path.exists(tsne_emb_path):
                    embdf = __import__('pandas').read_csv(tsne_emb_path)
                    emb = embdf[[col for col in embdf.columns if col.startswith('dim_')]].values
                    _save_scatter(emb, hier_labels, os.path.join(out_dir, "tsne_hier_scatter.png"), "t-SNE colored by Hierarchical")
            except Exception:
                pass

        except Exception:
            print("Plotting dependencies missing or failed; scatter plots skipped.")

        # Consolidate metrics
        try:
            import pandas as pd
            summary = []
            def _row(name, params, metrics):
                row = {"method": name, "params": str(params)}
                row.update(metrics)
                return row

            summary.append(_row("KMeans", {"k": k}, k_metrics))
            try:
                summary.append(_row("DBSCAN", {"eps": 0.5, "min_samples": 5}, db_metrics))
            except Exception:
                summary.append(_row("DBSCAN", {"eps": 0.5, "min_samples": 5}, {}))
            try:
                summary.append(_row("Hierarchical", {"n_clusters": k, "linkage": "ward"}, h_metrics))
            except Exception:
                summary.append(_row("Hierarchical", {"n_clusters": k, "linkage": "ward"}, {}))

            summary_df = pd.DataFrame(summary)
            summary_csv = os.path.join(out_dir, "summary_metrics.csv")
            summary_df.to_csv(summary_csv, index=False)
            print("Summary metrics written:", summary_csv)
        except Exception:
            print("Failed to write summary metrics:")
            traceback.print_exc()

        print("Done.")

    except Exception:
        print("Error during execution:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
