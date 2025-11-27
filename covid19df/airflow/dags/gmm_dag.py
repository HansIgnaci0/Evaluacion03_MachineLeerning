from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "retries": 0,
}

with DAG(
    dag_id="kedro_gmm_dag",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    run_pipeline = BashOperator(
        task_id="run_gmm_pipeline",
        bash_command="cd /opt/airflow/covid19df && kedro run --pipeline gmm",
    )
