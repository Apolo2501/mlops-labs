from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def hello_world():
    print("Hola Antonio, Airflow está ejecutando tu código!")

with DAG(
    dag_id="hello_world",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    tarea = PythonOperator(
        task_id="saludo",
        python_callable=hello_world
    )
