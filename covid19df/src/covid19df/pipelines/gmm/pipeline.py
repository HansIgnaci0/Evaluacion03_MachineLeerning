from kedro.pipeline import Pipeline, node
from .nodes import load_features, run_gmm, plot_gmm, gmm_metrics


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(func=load_features, inputs="df1", outputs="X_features_gmm", name="gmm_load_features"),
        node(func=run_gmm, inputs=["X_features_gmm", "params:gmm"], outputs="gmm_labels", name="gmm_run"),
        node(func=plot_gmm, inputs=["X_features_gmm", "gmm_labels"], outputs="gmm_plot", name="gmm_plot"),
        node(func=gmm_metrics, inputs=["X_features_gmm", "gmm_labels"], outputs="gmm_metrics", name="gmm_metrics"),
    ])
