from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'hansi',
    'depends_on_past': False,
    'email': ['tu_email@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'kedro_reduccion_dimensional_dag',
    default_args=default_args,
    description='DAG para ejecutar el pipeline de reducción de dimensionalidad (PCA + t-SNE)',
    schedule_interval=None,
    start_date=datetime(2025, 10, 22),
    catchup=False,
    tags=['kedro', 'reduccion_dimensional'],
)

run_reduccion = BashOperator(
    task_id='run_kedro_reduccion_dimensional',
    bash_command=(
        'cd /opt/airflow/covid19df && '
        'kedro run --pipeline reduccion_dimensional'
    ),
    dag=dag,
)

run_reduccion
