from kedro.pipeline import Pipeline, node
from .nodes import load_and_prepare_data, train_models_gridsearch

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_and_prepare_data,
                inputs=["params:covid_csv", "params:usa_csv"],
                outputs=["X", "y"],
                name="load_and_prepare_data_node"
            ),
            node(
                func=train_models_gridsearch,
                inputs=["X", "y"],
                outputs=["regression_comparison", "regression_comparison_plot"],
                name="train_models_gridsearch_node"
            ),
        ]
    )
