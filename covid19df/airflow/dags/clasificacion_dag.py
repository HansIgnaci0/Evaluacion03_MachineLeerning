from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Configuración del DAG (similar al de regresión)
default_args = {
    'owner': 'hansi',
    'depends_on_past': False,
    'email': ['tu_email@example.com'],  # Opcional: cambia a tu correo
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'kedro_clasificacion_dag',
    default_args=default_args,
    description='DAG para ejecutar el pipeline de clasificación de Kedro',
    schedule_interval=None,
    start_date=datetime(2025, 10, 22),
    catchup=False,
    tags=['kedro', 'clasificacion'],
)

# Tarea para ejecutar el pipeline de clasificación
run_clasificacion = BashOperator(
    task_id='run_kedro_clasificacion',
    bash_command=(
        'cd /opt/airflow/covid19df && '
        'kedro run --pipeline clasificacion'
    ),
    dag=dag
)

run_clasificacion
