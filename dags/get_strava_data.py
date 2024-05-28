from airflow import DAG
import shutil
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
import os
from airflow.models import Variable
import logging
from airflow.operators.dummy_operator import DummyOperator

def split_lat_lng(lat_lng_str):
    if isinstance(lat_lng_str, list) and not lat_lng_str:  # If lat_lng_str is an empty list
        return np.nan, np.nan  # Assign NaN or any other suitable value
    elif isinstance(lat_lng_str, str):  # If lat_lng_str is a string
        return lat_lng_str.split(',')
    else:
        return np.nan, np.nan


# Define default_args for the DAG
default_args = {
    'owner': 'Jackie',
    'depends_on_past': False,
    'start_date': datetime.now() - timedelta(days=7),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Instantiate the DAG
dag = DAG(
    dag_id='get_strava_data',
    default_args=default_args,
    description='A DAG to retrieve data from my strava API  weekly',
    schedule= '0 14 * * Mon')

def get_new_strava_data(ti):
    auth_url = 'https://www.strava.com/oauth/token'


    payload = {
        'client_id': '114109',
        'client_secret': 'dfb09adb989a2aceae8a6424a356babfd64879f1',
        'refresh_token': '751f49a40c83a88fa15ac119b8cff27c8cff9bbc',
        'grant_type': 'refresh_token'
    }

    print("Requesting Token..\n")

    res = requests.post(auth_url, data=payload,verify=True)


    all_activities = []

    #Make sure request was successful
    if res.status_code == 200:
        print('Request successful')
        refreshToken = res.json()['access_token']

        url = 'https://www.strava.com/api/v3/athlete/activities'
        header = {'Authorization': 'Bearer ' + refreshToken}
        request_page_num = 1
        startdate = (datetime.now() - timedelta(days=7)).timestamp()
        #end_date= datetime(2024,3,25).timestamp()
        


        while True:
            parm = {'per_page': 200,'page':request_page_num,'after':startdate}
            my_dataset = requests.get(url,headers=header,params=parm).json()
            if len(my_dataset)==0:
                break
            if all_activities:
                all_activities.extend(my_dataset)

            else:
                all_activities = my_dataset
            request_page_num +=1



    else:
        print(f"Token request failed with status code {res.status_code}")
        print(res.text)

    

    #Columns of interest
        #Columns of interest
    selected_fields = ['name', 'distance','moving_time', 'elapsed_time','gear_id', 'total_elevation_gain', 'type', 'sport_type', 'average_speed','max_speed', 'summary_polyline','start_date','timezone','elev_high','elev_low','average_heartrate', 'max_heartrate','average_cadence','start_latlng','end_latlng']
    summary_polyline = [activity['map']['summary_polyline'] for activity in all_activities]


    df = pd.DataFrame([{field: activity.get(field) for field in selected_fields} for activity in all_activities])
    df['summary_polyline'] = summary_polyline
    #convert gear_id to string
    
   
    df['gear_id'] = df['gear_id'].astype(str)
    if 'start_latlng' in df.columns:
        # Splitting 'start_latlng' into separate columns for latitude and longitude
        df['start_lat'], df['start_lng'] = zip(*df['start_latlng'].apply(split_lat_lng))
        # Dropping the original 'start_latlng' column
        df.drop(columns=['start_latlng'], inplace=True)

    if 'end_latlng' in df.columns:
        # Splitting 'end_latlng' into separate columns for latitude and longitude
        df['end_lat'], df['end_lng'] = zip(*df['end_latlng'].apply(split_lat_lng))
        # Dropping the original 'end_latlng' column
        df.drop(columns=['end_latlng'], inplace=True)
  
    ti.xcom_push(key='new_data', value=df)
  

def clean_data(ti):
    #pull in new data
    df = ti.xcom_pull(task_ids='get_strava_data', key='new_data')
   
    #shorten date
    df['start_date'] = pd.to_datetime(df['start_date'])

    # Extract the date part as a string
    df['start_date'] = df['start_date'].dt.strftime("%Y-%m-%d")

    #convert distance in meters to miles
    df['distance'] = (df['distance'].fillna(0) * 0.000621371).astype(float)

    #time is in s,coverting to mins
    df['moving_time'] = df['moving_time']/60
    df['elapsed_time'] = df['elapsed_time']/60

    # get the hours and mins
    df['moving_hours'], df['moving_mins'] = divmod(df['moving_time'] , 60)
    df['elapsed_hours'], df['elapsed_mins'] = divmod(df['elapsed_time'] , 60)

    #total time in mins
    df['total_moving_time'] = df['moving_hours']*60 + df['moving_mins']
    df['total_elapsed_time'] = df['elapsed_hours']*60 + df['elapsed_mins']

    #Speed is in m/s, coverting to  miles/hour
    df['average_speed'] = round((df['average_speed'] * 2.23694),1)
    df['max_speed'] = round((df['max_speed'] * 2.23694),1)

    # defining average pace per miles
    df['pace'] = df['total_elapsed_time']/df['distance']

    #Get mins and seconds for mile pace
    df['average_pace_mins'] ,df['average_pace_secs']=  divmod(df['pace']*60,60)
    # Round to prevent infinite numbers
    df['average_pace_mins'] = df['average_pace_mins'].round()
    df['average_pace_secs'] = df['average_pace_secs'].round()
   
    #Replace infinite,nan values
    df = df.replace([np.inf, -np.inf, np.nan], 0)
    df['average_pace_mins'] = df['average_pace_mins'].astype(int)
    df['average_pace_secs'] = df['average_pace_secs'].astype(int)

    #pad seconds
    df['average_pace_secs'] = df['average_pace_secs'].astype(str).str.zfill(2)
    #Calculate the average mins and second pace
    df['average_pace'] = df['average_pace_mins'].astype(str) + ':' + df['average_pace_secs']
    df = df.drop(labels=['pace', 'average_pace_mins','average_pace_secs'], axis=1)
    ti.xcom_push(key='cleaned_data', value=df)

def concatenate_data(ti):
    #Read the uploaded file
    print("Files in directory:")
    print(os.listdir('/opt/airflow/dags'))

    # Read the uploaded file
    file_path = '/opt/airflow/dags/Strava_update.csv'
    df_uploaded = pd.read_csv(file_path)
    # print(len(df_uploaded))
    # Retrieve the new Strava data from XCom
    df_strava = ti.xcom_pull(task_ids='clean_new_data', key='cleaned_data')
   
    # Concatenate the data
    df_updated = pd.concat([df_strava, df_uploaded], ignore_index=True)
    
    target_directory = '/opt/airflow/dags'

    # Save the concatenated dataframe to a CSV file
    df_updated.to_csv(os.path.join(target_directory, 'Strava_update.csv'), index=False)


task_1 = PythonOperator(
    task_id='get_strava_data',
    python_callable=get_new_strava_data,
    dag=dag,
)


task_2 = PythonOperator(
    task_id='clean_new_data',
    python_callable=clean_data,
    dag=dag,
)

task_3= PythonOperator(
    task_id='concatenate_data',
    python_callable=concatenate_data,
    dag=dag,
)

task_1 >> task_2 >> task_3