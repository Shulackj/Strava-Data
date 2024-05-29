import requests
import pandas as pd
import json


# Base URL of the WeatherAPI API
base_url = 'http://api.weatherapi.com/v1/'

# Endpoint to get current weather data for a specific location
endpoint = '/history.json'

# races = {'hyner':'hyner','eastern':'little_pine','ES':'little_pine'}

# locations = {'location':{'canoe_creek':{'lat': '40.48031', 'long':'-78.2913'},'worlds_end':{'lat':'41.4718','long':'-76.58145'},'little_pine':{'lat':'41.6354', 'long':'-77.35740'}, 
                    #     'hyner':{'lat':'41.35837','long':'-77.6281'}}}


# Parameters for the API request (e.g., location)
api_key = "e1eaaef502234394acf141157242903"
# latitude = locations['little_pine']['lat']
# longitude = locations['little_pine']['long']
date = "2023-08-12"

# Make the GET request to the API
url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={latitude},{longitude}&dt={date}"


response = requests.get(url)
data = response.json()

# Process the response data
print(data)

response = requests.get(url)


# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Extract and print the JSON data (weather information)
    weather_data = response.json()
    


    hourly_data = weather_data['forecast']['forecastday'][0]['hour']

    # Create an empty list to store the extracted data
    hourly_data_list = []

    # Iterate over each hour and extract the relevant information
    for hour in hourly_data:
        hour_data = {
            'time': hour['time'],
            'temp_f': hour['temp_f'],
            'condition_text': hour['condition']['text'],
            'humidity': hour['humidity'],
            'feelslike': hour['feelslike_f']
        }
        hourly_data_list.append(hour_data)

    # Convert the list of dictionaries into a DataFrame
    weather_df = pd.DataFrame(hourly_data_list)

    # Display the DataFrame
    print(weather_df)

 
  
else:
    # If the request was not successful, print the error status code
    print(f'Error: {response.status_code}')
