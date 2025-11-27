from kedro.pipeline import Pipeline, node
from .nodes import load_features, run_hierarchical, plot_hierarchical, jerarquico_metrics, jerarquico_dendrogram


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(func=load_features, inputs="df1", outputs="X_features_jerarquico", name="jerarquico_load_features"),
        node(func=run_hierarchical, inputs=["X_features_jerarquico", "params:jerarquico"], outputs="jerarquico_labels", name="jerarquico_run"),
        node(func=plot_hierarchical, inputs=["X_features_jerarquico", "jerarquico_labels"], outputs="jerarquico_plot", name="jerarquico_plot"),
        node(func=jerarquico_metrics, inputs=["X_features_jerarquico", "jerarquico_labels"], outputs="jerarquico_metrics", name="jerarquico_metrics"),
        node(func=jerarquico_dendrogram, inputs=["X_features_jerarquico", "params:jerarquico"], outputs="jerarquico_dendrogram", name="jerarquico_dendrogram"),
    ])
