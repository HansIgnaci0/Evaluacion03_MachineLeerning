from kedro.pipeline import Pipeline, node
from .nodes import load_and_prepare_data_classif, train_classification_models_gridsearch

def create_classification_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=load_and_prepare_data_classif,
                inputs=["params:covid_csv", "params:usa_csv"],
                outputs=["X_cls", "y_cls"],
                name="load_and_prepare_data_classif_node",
            ),
            node(
                func=train_classification_models_gridsearch,
                inputs=["X_cls", "y_cls"],
                outputs=["classification_comparison", "classification_comparison_plot"],
                name="train_classification_models_gridsearch_node",
            ),
        ]
    )


def create_pipeline(**kwargs):
    """Compatibilidad: Kedro espera `create_pipeline`.

    Wrapper que delega en `create_classification_pipeline` para mantener
    compatibilidad con `pipeline_registry`.
    """
    return create_classification_pipeline(**kwargs)
