# covid19df/airflow/dags/regresion_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Configuración del DAG
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
    'kedro_regresion_dag',
    default_args=default_args,
    description='DAG para ejecutar el pipeline de regresión de Kedro',
    schedule_interval=None,
    start_date=datetime(2025, 10, 22),
    catchup=False,
    tags=['kedro', 'regresion'],
)

# Tarea para ejecutar el pipeline de regresión
run_regresion = BashOperator(
    task_id='run_kedro_regresion',
    bash_command=(
        # Navega al proyecto
        'cd /opt/airflow/covid19df && '
        # Ejecuta Kedro pipeline de regresión
        'kedro run --pipeline regresion'
    ),
    dag=dag
)

run_regresion
