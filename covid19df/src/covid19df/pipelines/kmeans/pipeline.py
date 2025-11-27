from kedro.pipeline import Pipeline, node
from .nodes import load_features, run_kmeans, plot_kmeans, kmeans_metrics, kmeans_elbow


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(func=load_features, inputs="df1", outputs="X_features_kmeans", name="kmeans_load_features"),
        node(func=run_kmeans, inputs=["X_features_kmeans", "params:kmeans"], outputs="kmeans_labels", name="kmeans_run"),
        node(func=plot_kmeans, inputs=["X_features_kmeans", "kmeans_labels"], outputs="kmeans_plot", name="kmeans_plot"),
        node(func=kmeans_metrics, inputs=["X_features_kmeans", "kmeans_labels"], outputs="kmeans_metrics", name="kmeans_metrics"),
        node(func=kmeans_elbow, inputs=["X_features_kmeans", "params:kmeans"], outputs="kmeans_elbow_plot", name="kmeans_elbow"),
    ])
