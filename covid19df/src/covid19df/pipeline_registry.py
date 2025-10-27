from covid19df.pipelines import eda as eda_pipeline
from covid19df.pipelines import regresion as regresion_pipeline
from covid19df.pipelines.clasificacion import create_pipeline as create_classification_pipeline

def register_pipelines():
    return {
        "__default__": eda_pipeline.create_pipeline(),
        "eda": eda_pipeline.create_pipeline(),
        "regresion": regresion_pipeline.create_pipeline(),
        "clasificacion": create_classification_pipeline(),
    }
