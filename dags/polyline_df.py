import pandas as pd

df = pd.read_csv('/Users/jackie/Desktop/Strava//dags/Strava_update.csv')

df = df.drop_duplicates()

df['summary_polyline'] = df['summary_polyline'].replace('NaN', pd.NA)
df['distance'] = df['distance'].round(2)

updated_df = df.dropna(subset=['summary_polyline'])


updated_df.to_csv('/Users/jackie/Desktop/Strava/leaflet/my-app/public/Strava_update.csv')


