from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args ={
    'owner':'coder2j',
    'retries':5,
    'retry_delay':timedelta(minutes=5)
}
with DAG(dag_id='dag_with_cron_exp_v03',
         default_args=default_args,
         start_date = datetime(2023,1,10),
         schedule='0 8 * * Mon',
         catchup=False
         ) as dag:
    task1 = BashOperator(
        task_id='task1',
        bash_command='echo dag with cron expression!'
    )