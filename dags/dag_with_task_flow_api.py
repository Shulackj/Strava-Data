from datetime import datetime, timedelta
from airflow.decorators import dag, task


default_args ={
    'owner':'coder2j',
    'retries':5,
    'retry_delay':timedelta(minutes=5)

}

@dag(dag_id='dag_with_task_flow_api_v02',
     default_args=default_args,
     start_date=datetime(2024,2,12),
     schedule='@daily'
    )
def hello_world_etl():

    @task(multiple_outputs=True)
    def get_name():
        return {'first_name':'Jerry',
                'last_name':"Smith"}
    @task()
    def get_age():
        return 20
    @task() 
    def greet(first_name,last_name,age):
        print(f"hello world! My name is {first_name} {last_name} and I am {age} years old")
    name_dic = get_name()
    age = get_age()
    greet(first_name=name_dic['first_name'],last_name=name_dic['last_name'],age=age)

greet_dag = hello_world_etl()