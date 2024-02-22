import requests
from datetime import datetime,timedelta
import pandas as pd
import numpy as np


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
    #startdate = (datetime(2023,8,12)).timestamp()

    
    end_date = datetime(2024,2,5).timestamp()

    while True:
        parm = {'per_page': 200,'page':request_page_num,'before':end_date}
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


#Preview the response
print(all_activities[0])

#Current Columns
print(df.columns)

#Columns of interest
selected_fields = ['name', 'distance','moving_time', 'elapsed_time','gear_id', 'total_elevation_gain', 'type', 'sport_type', 'average_speed','max_speed', 'summary_polyline','start_date','timezone','elev_high','elev_low','average_heartrate', 'max_heartrate','average_cadence','notes']

#create a df
df = pd.DataFrame([{field: activity.get(field) for field in selected_fields} for activity in all_activities])



def clean_data(df):
    #distance is in m, coverting to miles
    df['distance'] = (df['distance'] * 0.000621371).astype(float)

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
    return(df)


df=clean_data(df)

#Drop intermediate columns
df = df.drop(labels=['pace', 'average_pace_mins','average_pace_secs'], axis=1)

#Write to file
df.to_csv('Strava20240205.csv', index=False)






