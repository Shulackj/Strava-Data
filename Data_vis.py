import pandas as pd
import streamlit as st
import numpy as np
import cufflinks as cf
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
cf.go_offline()
import matplotlib .pyplot as plt
from datetime import datetime,timedelta
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.figure_factory import create_table
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#read in strava data
df = pd.read_csv('/workspaces/Strava-Data/dags/Strava_update.csv')

#print the len
#print(len(df))
#print(df.dtypes)
#drop duplicates
df= df.drop_duplicates()
#print(len(df))

#Aligning activities
df['type'] = df['type'].replace({'VirtualRide': 'Ride','Workout':'WeightTraining'})

df['date'] = np.array(pd.to_datetime(df['start_date']))
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['starting_week'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='D')
#print(df['name'].unique())
trail_df = df[df['name'].str.contains('trail', case=False, na=False)]
trail_df = trail_df[['name','type','sport_type']]

#print(df['sport_type'].unique())
#Streamlit app
#page navigation
st.sidebar.title('Data Overview')
selected_page = st.sidebar.radio("Select Page", ['Current','Races','All Time', 'Yearly Breakdown'])

    
#Function to return stacked bar charts
def stacked_bar_chart(data_frame,x,y,labels,title,orientation):
    chart = px.bar(data_frame, x=x, y=y, color='type',
    labels=labels,
    title=title, orientation=orientation)
    return chart
#Function to get the last 12 weeks of data
def get_last_3_months_data(date=None,df=df):
    if date is None:
        date = datetime.now()
    else:
        date=date

    most_recent_monday = date- timedelta(days=(date.weekday() + 6) % 7)
    # Calculate the start date for the last 12 weeks from the most recent Monday
    start_date = most_recent_monday - pd.DateOffset(weeks=12)
    # Generate a range of dates for the last 12 weeks
    date_range_last_12_weeks = pd.date_range(start=start_date, end=most_recent_monday, freq='W')
    # Filter the DataFrame for rows within the last 12 weeks
    df = df[(df['starting_week'] >= start_date) & (df['starting_week'] <= date)]
    return df

def volume_calculator(df):
    df = df.groupby('starting_week')['total_elapsed_time'].sum().reset_index()
    df['total_elapsed_time'] = df['total_elapsed_time'].apply(lambda x: "%d:%02d" % (divmod(x, 60)))
    df['starting_week'] = df['starting_week'].dt.strftime('%m-%d')
    df = df.rename(columns={'starting_week': 'Date', 'total_elapsed_time': 'Total Time'})
    return df

def distance_calc(df):
    grouped_df = df.groupby(['type', pd.Grouper(key='starting_week', freq='W-MON')])[['distance','total_elapsed_time']].sum().reset_index()
    grouped_df = grouped_df.sort_values('starting_week')
    return grouped_df


if selected_page == 'Current':
    # Set the 'date' column as the index
    df_time = df
    #Get Rid of data without distance
    df_distance = df[(df['type'] != 'Elliptical') & (df['type'] != 'WeightTraining')]

    #Get weekly distance and duration totals
    df_distance_calc = distance_calc(df_distance)
    ltm_df = get_last_3_months_data(date=None,df=df_distance_calc)

    last_3_months = stacked_bar_chart(ltm_df,x='starting_week',y='distance',labels={'starting_week':'Date','distance':'Distance in Miles'},title='Weekly Distance Totals by Activity',orientation='v')
    last_3_months.update_xaxes(tickmode='array', tickvals=ltm_df['starting_week'], ticktext=ltm_df['starting_week'].dt.strftime('%m-%d'))
    st.plotly_chart(last_3_months)

    weekly_dur = df_time.groupby(['type', pd.Grouper(key='starting_week', freq='W-MON')])[['total_elapsed_time']].sum().reset_index()
    weekly_dur = weekly_dur.sort_values('starting_week')

    ltm_time = get_last_3_months_data(date=None,df=weekly_dur)
   
    ltm_time['updated_time'] = ltm_time['total_elapsed_time'].apply(lambda x: "%d:%02d" % (divmod(x, 60)))

    last_three_dur = stacked_bar_chart(ltm_time,y='starting_week',x='total_elapsed_time',labels={'starting_week':'Date','total_elapsed_time':'Time in Mins.'},title='Weekly Duration Totals by Activity',orientation='h')

    last_three_dur.update_yaxes(tickmode='array', tickvals=ltm_time['starting_week'], ticktext=ltm_time['starting_week'].dt.strftime('%m-%d'))
    st.plotly_chart(last_three_dur)

    #total aerobic volume
    aerobic_vol = ltm_time[ltm_time['type']!='WeightTraining']

    #total cross train
    xtrain_vol = ltm_time[(ltm_time['type'] == 'Ride') | (ltm_time['type'] == 'Hike') | (ltm_time['type'] == 'Elliptical')]

    #total run
    run_vol = ltm_time[ltm_time['type']=='Run']

    df1 = volume_calculator(aerobic_vol)
    df2 = volume_calculator(xtrain_vol)
    df3 = volume_calculator(run_vol)

    merged_df = pd.merge(df1, df2, on='Date', how='inner')
    merged_df = merged_df.rename(columns={'Total Time_x':'Total Aerobic Time','Total Time_y':'Total Cross Train Time'})
    final_vol = pd.merge(merged_df, df3, on='Date', how='inner')
    final_vol = final_vol.rename(columns={'Total Time':'Total Run Time'})
    
    #Create table
    fig = ff.create_table(final_vol)
  
    st.plotly_chart(fig)



if selected_page == 'Races':
    def get_countdown(end_time):
        current_time = datetime.now()
        time_difference = end_time - current_time
        return time_difference

    # Display countdown
    end_time = datetime(2024, 5, 18, 00, 5, 00)
    countdown = get_countdown(end_time)
    days = countdown.days
    hours = countdown.seconds // 3600  
    minutes = (countdown.seconds % 3600) // 60 

    st.title("Countdown to Next Race")
    st.write(f'Days until next race: {days} days, {hours} hours, {minutes} minutes')


    race_types = ['50k','100k','100 Miles']

    # Create a dropdown for selecting race type
    selected_race_type = st.selectbox('Analyze Past Races', race_types)
    if selected_race_type == '50k':
        hyner_df = df[df['name']=='Hyner 50k']
        dates = hyner_df['date'].unique()

        selected_year = st.selectbox('Select a Year', sorted(hyner_df['date'].dt.year.unique()))

        filtered_df = hyner_df[hyner_df['date'].dt.year == selected_year]
        
        selected_date_index = None
        for i, date in enumerate(dates):
            if date.year == selected_year:
                ind = i
                break
        
        ltm_hyner = get_last_3_months_data(date=dates[ind],df=df)

        hyner_distance = distance_calc(ltm_hyner)
        
        last_three_hyner = stacked_bar_chart(hyner_distance,x='starting_week',y='distance',labels={'starting_week':'Date','distance':'Distance'},title='Weekly Duration Totals by Activity',orientation='v')

        last_three_hyner.update_xaxes(tickmode='array', tickvals=hyner_distance['starting_week'], ticktext=hyner_distance['starting_week'].dt.strftime('%m-%d'))
        st.plotly_chart(last_three_hyner)

        hyner_dur = volume_calculator(ltm_hyner)
        print(hyner_dur)
        # last_three_hyner_dur = stacked_bar_chart(hyner_distance,y='Date',x='total_elasped_time',labels={'Date':'Date','total_elapsed_time':'Total Time in Mins'},title='Weekly Duration Totals by Activity',orientation='h')

        # last_three_hyner_dur.update_yaxes(tickmode='array', tickvals=hyner_dur['Date'], ticktext=hyner_dur['Date'].dt.strftime('%m-%d'))
        # st.plotly_chart(last_three_hyner_dur)

if selected_page == 'All Time':

#calculate total distnaes per activity type per year
    total_distance_per_year = df[df['type'] != 'Elliptical']
    total_distance_per_year = total_distance_per_year.groupby(['type', 'year'])['distance'].sum().reset_index()
    
    st.title("Distance by Year and Activity")

    # Show the figure
    distance_by_type = stacked_bar_chart(total_distance_per_year,x='year',y='distance',labels={'distance':'Total Distance (Miles)','year':'Year'}, title='Annual Distance Summary by Activity')
    st.plotly_chart(distance_by_type) 

    # total time per activity per year 
    total_time_per_year = df.groupby(['type', 'year'])['total_elapsed_time'].sum().reset_index()
    
    st.title("Toal Duration by Year and Activity")

    duration_by_type = stacked_bar_chart(total_time_per_year,x='year',y='total_elapsed_time',labels={'total_elapsed_time':'Total Time (Minutes)','year':'Year'}, title='Annual Duration Summary by Activity')


    # Show the figure
    st.plotly_chart(duration_by_type) 


    #Percent of runs on road vs. trail
    run_type = df[df['sport_type'].isin(['Run', 'TrailRun'])]
    all_runs = run_type.groupby(['year'])['distance'].sum().reset_index()
    run_type = run_type.groupby(['sport_type','year'])['distance'].sum().reset_index()
    
    # print(all_runs)
    # print(run_type)

    merged_df = pd.merge(all_runs, run_type, on='year', how='left')
    merged_df= merged_df.rename(columns={'distance_x':'Total Distance','distance_y':'run_type_distance'})
    # print(merged_df)

    # percent_trail= merged_df[merged_df['sport_type']] / all_runs) * 100
    # percent_road= (run_type[run_type['sport_type']=='Run'] / all_runs) * 100
    # print(percent_trail)
    merged_df['percent'] = ((merged_df['run_type_distance'] / merged_df.groupby('year')['Total Distance'].transform('sum')) * 100).round()
    # print(merged_df)
    

# Create a pie chart with rounded percentages
    trail_to_road =  px.pie(merged_df, names='sport_type', values='percent', 
                title='Trail Running vs Road Running per Year', 
                color='sport_type', 
                facet_col='year',
                labels={'percent': 'Percentage of Total Miles Run'})

    # Show the chart
    st.plotly_chart(trail_to_road)

if selected_page == 'Yearly Breakdown':
    st.title('Activites by Year')
    # Year selection button
    selected_year = st.selectbox('Select a Year', sorted(df['date'].dt.year.unique()))

    # Filter data based on selected year
    filtered_df = df[df['date'].dt.year == selected_year]

    # Plot the data
    fig_yearly = px.bar(filtered_df, x='date', y='total_elapsed_time', title=f'Data for {selected_year}')


    fig_yearly.update_layout(xaxis_title='Month', yaxis_title='Total Time',
                            xaxis=dict(tickmode='array', tickvals=pd.date_range(start=filtered_df['date'].min(), end=filtered_df['date'].max(), freq='M').tolist(), tickformat='%b'))

    st.plotly_chart(fig_yearly)

    #get monthly totals
    monthly_totals = filtered_df.resample('M', on='date').sum()

    # Plot the monthly breakdown
    fig_monthly = px.bar(monthly_totals, x=monthly_totals.index, y='total_elapsed_time', labels={'Value': 'Monthly Total'},
                        title=f'Monthly Totals for  {selected_year}')

    fig_monthly.update_layout(xaxis_title='Month', yaxis_title='Total Time',
                            xaxis=dict(tickmode='array', tickvals=pd.date_range(start=filtered_df['date'].min(), end=filtered_df['date'].max(), freq='M').tolist(), tickformat='%b'))
    # Display the monthly plot

    st.plotly_chart(fig_monthly)



    #stack by type of activity
    st.title("Time Spend by Activity")


    # Create stacked bar chart
    fig_by_type = px.bar(filtered_df, x='month', y='total_elapsed_time', color='type',
                labels={'total_elapsed_time': 'Total Time (minutes)', 'month': 'Month'},
                title='Stacked Bar Chart of Time Spent by Activity',
                category_orders={'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})

    # Show the figure
    st.plotly_chart(fig_by_type) 



