def register_pipelines():
    """Registrar pipelines de forma segura (imports perezosos).

    Evita importar módulos que a su vez requieran `kedro` en el import-time.
    Cada pipeline se importa dentro de un try/except para que la falta de
    dependencias en el entorno no impida que este archivo se importe y que
    otros pipelines se registren.
    """
    pipelines = {}

    # Helper para intentar importar un pipeline y añadirlo al dict
    def try_register(key, import_path, create_name="create_pipeline"):
        try:
            module = __import__(import_path, fromlist=[create_name])
            creator = getattr(module, create_name)
            pipelines[key] = creator()
            return True
        except Exception as exc:  # pragma: no cover - runtime diagnostic
            import traceback
            import sys

            traceback.print_exc()
            print(f"Warning: failed to import/register '{key}': {exc}", file=sys.stderr)
            return False

    # Registrar pipelines principales de forma perezosa
    try_register("eda", "covid19df.pipelines.eda")
    try_register("regresion", "covid19df.pipelines.regresion")
    # Clasificación usa un nombre de creador distinto en algunos módulos
    try_register("clasificacion", "covid19df.pipelines.clasificacion", create_name="create_pipeline")

    # Pipelines opcionales/experimentales
    try_register(
        "clustering_noSupervisado",
        "covid19df.pipelines.unsupervised.clustering",
        create_name="create_pipeline",
    )
    try_register("reduccion_dimensionalidad", "covid19df.pipelines.unsupervised.reduction")

    # Definir __default__ de forma segura (prefiere eda, luego regresion, luego clasificacion)
    pipelines["__default__"] = (
        pipelines.get("eda") or pipelines.get("regresion") or pipelines.get("clasificacion") or {}
    )

    return pipelines
