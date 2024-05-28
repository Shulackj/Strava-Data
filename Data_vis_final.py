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
df = pd.read_csv('/Users/jackie/Desktop/Strava//dags/Strava_update.csv')

#print the len
print(len(df))
print(df.columns)

#print(df.dtypes)
#drop duplicates
df = df.drop_duplicates()
print(len(df))

#Aligning activities
df['type'] = df['type'].replace({'VirtualRide': 'Ride','Workout':'WeightTraining','Yoga':'Sauna'})

df['date'] = np.array(pd.to_datetime(df['start_date']))
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['starting_week'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='D')
#print(df['name'].unique())
trail_df = df[df['name'].str.contains('trail', case=False, na=False)]
trail_df = trail_df[['name','type','sport_type']]
df_50k = df[df['name']=='Hyner 50k']

#print(df['sport_type'].unique())
#Streamlit app
#page navigation
st.sidebar.title('Data Overview')
selected_page = st.sidebar.radio("Select Page", ['Current','Races','Compare Training','All Time', 'Yearly Breakdown','Map'])

    
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
    df['week_number'] = df['starting_week'].dt.strftime('Week %U')
    return df
def get_data_between_dates(start_date=None, end_date=None, df=None):
     # Filter the DataFrame for rows within the specified date range
    df = df[(df['starting_week'] >= start_date) & (df['starting_week'] <= end_date)]
    df['week_number'] = df['starting_week'].dt.strftime('Week %U')
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

def get_race_data(race_df,df,selected_year):
    dates = race_df['date'].unique()
    filtered_df = race_df[race_df['date'].dt.year == selected_year]
    
    selected_date_index = None
    for i, date in enumerate(dates):
        if date.year == selected_year:
            ind = i
            break
    
    ltm_df = get_last_3_months_data(date=dates[ind],df=df)

    ltm_distance = distance_calc(ltm_df)

    return ltm_df,ltm_distance

def assign_week_number(df):
    # Sort the DataFrame by the 'starting_week' column
    df_sorted = df.sort_values(by='starting_week')
    
    # Create a new column to store the week number
    df_sorted['week_num'] = 1
    
    # Assign a unique week number to each starting week
    current_week_number = 0
    previous_week = None
    for index, row in df_sorted.iterrows():
        if row['starting_week'] != previous_week:
            current_week_number += 1
        previous_week = row['starting_week']
        df_sorted.at[index, 'week_num'] = current_week_number
    
    return df_sorted
def finish_times(df):
    df['Finishing_time'] = df['total_elapsed_time'].apply(lambda x: "%d:%02d" % (divmod(x, 60)))
    df = df.sort_values(by='total_elapsed_time')

    fig = px.scatter(df, x='date', y='Finishing_time', title='Finishing Times for Past Years',
                 color=np.where(df['finish'] == 'DNF', 'DNF', 'Finished'),
                 color_discrete_map={'DNF': 'red', 'Finished': 'purple'},
                 labels={'Finishing_time': 'Finishing Time'})

    # Show legend
    fig.update_traces(marker=dict(size=10, symbol='circle'), showlegend=True)
    fig.update_layout(legend=dict(title='Race Result'))
    
    fig.update_xaxes(
    title='Year',
    tickvals=['2018','2019','2020','2021','2022','2023','2024'],  # Specify the years
    ticktext=['2018','2019','2020','2021','2022','2023','2024'],  # Specify the year labels
    range=['2018','2024']  # Specify the desired range for x-axis
)       
    fig.update_yaxes(title='Finishing Time')
    return fig

def decode_polyline(polyline_str):
    if polyline_str:
        coords = polyline.decode(polyline_str)
        lats, lons = zip(*coords)
        return list(lats), list(lons)  # Convert tuples to lists
    else:
        return [], []
 

if selected_page == 'Current':
    st.markdown("## :bar_chart: Current Training Analysis", unsafe_allow_html=True)
    st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
    st.markdown('<style>div.block-container{padding-bottom:2rem;}</style>', unsafe_allow_html=True)

    # Set the 'date' column as the index
    df_time = df
    #Get Rid of data without distance
    df_distance = df[(df['type'] != 'Elliptical') & (df['type'] != 'WeightTraining')& (df['type'] != 'Sauna')]

    #Get weekly distance and duration totals
    df_distance_calc = distance_calc(df_distance)
    ltm_df = get_last_3_months_data(date=None,df=df_distance_calc)

    last_3_months = stacked_bar_chart(ltm_df,x='starting_week',y='distance',labels={'starting_week':'Date','distance':'Distance in Miles'},title='Weekly Distance Totals by Activity',orientation='v')
    last_3_months.update_xaxes(tickmode='array', tickvals=ltm_df['starting_week'], ticktext=ltm_df['starting_week'].dt.strftime('%m-%d'))
    last_3_months.update_layout(legend=dict(title='Activity'))
    st.plotly_chart(last_3_months)

    weekly_dur = df_time.groupby(['type', pd.Grouper(key='starting_week', freq='W-MON')])[['total_elapsed_time']].sum().reset_index()
    weekly_dur = weekly_dur.sort_values('starting_week')

    ltm_time = get_last_3_months_data(date=None,df=weekly_dur)
   
    ltm_time['updated_time'] = ltm_time['total_elapsed_time'].apply(lambda x: "%d:%02d" % (divmod(x, 60)))

    last_three_dur = stacked_bar_chart(ltm_time,y='starting_week',x='total_elapsed_time',labels={'starting_week':'Date','total_elapsed_time':'Time in Mins.'},title='Weekly Duration Totals by Activity',orientation='h')

    last_three_dur.update_yaxes(tickmode='array', tickvals=ltm_time['starting_week'], ticktext=ltm_time['starting_week'].dt.strftime('%m-%d'))
    last_three_dur.update_xaxes(showgrid=True, gridcolor='DarkSlateBlue')
    last_three_dur.update_layout(legend=dict(title='Activity'))

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
    ccolorscale = [[0, '#42435B'],[.5, '#8386B3'],[1, '#A6A9DD']]
    
    fig = ff.create_table(final_vol, colorscale=ccolorscale, font_colors=['white'])
  
    st.plotly_chart(fig)


# start_date = datetime(2024, 1, 1)  # January 1, 2024
# end_date = datetime(2024, 4, 1) 
# compare_df = get_data_between_dates(start_date=start_date, end_date=end_date, df=df)
#     #compare_df =  assign_week_number(compare_df)
# selected_year = compare_df[['year']]
# compare_dur =  volume_calculator(compare_df)
# print(compare_dur)



if selected_page == 'Races':
    st.markdown("## :woman-running: Race Analysis", unsafe_allow_html=True)
    st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
    st.markdown('<style>div.block-container{padding-bottom:2rem;}</style>', unsafe_allow_html=True)

    def get_countdown(end_time):
        current_time = datetime.now()
        time_difference = end_time - current_time
        return time_difference

    # Display countdown
    end_time = datetime(2024, 8, 10, 00, 5, 00)
    countdown = get_countdown(end_time)
    days = countdown.days
    hours = countdown.seconds // 3600  
    minutes = (countdown.seconds % 3600) // 60 

    st.write(f":clock1: Countdown until next race: {days} days, {hours} hours, {minutes} minutes")
    st.markdown('<style>div.block-container{padding-bottom:4rem;}</style>', unsafe_allow_html=True)

    race_types = ['50k','100k','100 Miles']
    
    # df_100 = df[df['name'].str.contains('ES') |df['name'].str.contains('Eastern')  ]
    # print(df_100)
    # Create a dropdown for selecting race type
    races = st.sidebar.selectbox('Pick a Race Distance',race_types)
    
    if not races:
        df2=df.copy()
    elif races == '50k':
        df_50k = df[df['name']=='Hyner 50k']    
        df_50k['finish']=  'completed race'  
        st.plotly_chart(finish_times(df_50k))  
        
        dates = df_50k['date'].unique()
        selected_year = st.selectbox('Select a Year', sorted(df_50k['date'].dt.year.unique()))

        filtered_df = df_50k[df_50k['date'].dt.year == selected_year]
        
        selected_date_index = None
        for i, date in enumerate(dates):
            if date.year == selected_year:
                ind = i
                break
        
        ltm_hyner = get_last_3_months_data(date=dates[ind],df=df)

        hyner_distance = distance_calc(ltm_hyner)

        ltm_hyner,hyner_distance = get_race_data(df_50k,df,selected_year)
        
        last_three_hyner = stacked_bar_chart(hyner_distance,x='starting_week',y='distance',labels={'starting_week':'Date','distance':'Distance'},title=f'Weekly Duration Totals by Activity for {selected_year}',orientation='v')

        last_three_hyner.update_xaxes(tickmode='array', tickvals=hyner_distance['starting_week'], ticktext=hyner_distance['starting_week'].dt.strftime('%m-%d'))
        st.plotly_chart(last_three_hyner)

        hyner_dur = volume_calculator(ltm_hyner)
        #print(hyner_dur)

    elif races == '100k':
        df_100k = df[df['name'].str.contains('World') |df['name'].str.contains('Iron')|df['name'].str.contains('Black') ]
        df_100k['finish'] = np.where(df_100k['name'].str.contains('DNF'), 'DNF', 'completed race')

        st.plotly_chart(finish_times(df_100k))
    elif races == '100 Miles':
        df_100 = df[df['name'].str.contains('ES') |df['name'].str.contains('Eastern') |df['name'].str.contains('MMT')  ]
        df_100['finish'] = np.where(df_100['name'].str.contains('DNF'), 'DNF', 'completed race')

        st.plotly_chart(finish_times(df_100))

# # Create the Streamlit selectbox with the first year as the default value
#             #Add current year to 50k dates
#             date_list = list(df_50k['date'].dt.year.unique())
#             date_list.append(2024)
#             #print(date_list)

#             year_one =  st.selectbox('Select First Year', sorted(date_list))
#             year_two = st.selectbox("Select Second Year", sorted([x for x in date_list if x != year_one]))
            
#             if year_one == 2024:
#                 current_race = datetime(2024, 4, 20)
#                 df= get_last_3_months_data(date=current_race,df=df)
#                 hyner_distance_year_one =distance_calc(df)
#                 hyner_distance_year_one = assign_week_number(hyner_distance_year_one)
#             else:
#                 ltm_hyner_year_one,hyner_distance_year_one = get_race_data(df_50k,df,year_one)
#                 hyner_distance_year_one = assign_week_number(hyner_distance_year_one)
#             #hyner_distance_year_one['week_num'] = hyner_distance_year_one['starting_week'].apply(get_week_number(start_date=year_one_start,end_date=year_one_end))
#             if year_two == 2024:
#                 current_race = datetime(2024, 4, 20)
#                 df= get_last_3_months_data(date=current_race,df=df)
#                 hyner_distance_year_two =distance_calc(df)
#                 hyner_distance_year_two = assign_week_number(hyner_distance_year_two)

#             else:
#                 ltm_hyner_year_two,hyner_distance_year_two = get_race_data(df_50k,df,year_two)
#                 hyner_distance_year_two = assign_week_number(hyner_distance_year_two)


#             hyner_distance_year_one['year'] = year_one
#             hyner_distance_year_two['year'] = year_two

#         # Concatenate the two DataFrames
#         combined_df = pd.concat([hyner_distance_year_one, hyner_distance_year_two], ignore_index=True)
#         print(combined_df)
#         # Grouping by week, year, and activity type and aggregating data
#         grouped_data = combined_df.groupby(['week_num', 'year', 'type'])['distance'].sum().reset_index()
  

#          #Create comparsion figure
#         traces = []
#         years = grouped_data['year'].unique()
#         colors = {'Run': 'blue', 'Ride': 'green'}  # Add more colors if needed

#         for year in years:
#             data_year = grouped_data[grouped_data['year'] == year]
#             trace = go.Bar(
#                 x=data_year['week_num'],
#                 y=data_year['distance'],  # Adjust this to your specific data column
#                 name=str(year),  # Year as the trace name
#                 hoverinfo='text',
#                 text=data_year['type'],  # Display activity type on hover
#                 hoverlabel=dict(namelength=-1)  # Display full activity type name on hover
#             )
#             traces.append(trace)

#         # Creating layout
#         layout = go.Layout(
#             title='Comparison of Data by Week and Activity Type for Two Years',
#             xaxis=dict(title='Week'),
#             yaxis=dict(title='Total Distance'),  # Adjust this to your specific data column
#             barmode='group'  # Grouped bars
#         )

#         # Creating figure
#         fig = go.Figure(data=traces, layout=layout)

#         # Show the plot
#         st.plotly_chart(fig)
          


#         # last_three_hyner_dur = stacked_bar_chart(hyner_distance,y='Date',x='total_elasped_time',labels={'Date':'Date','total_elapsed_time':'Total Time in Mins'},title='Weekly Duration Totals by Activity',orientation='h')

        # last_three_hyner_dur.update_yaxes(tickmode='array', tickvals=hyner_dur['Date'], ticktext=hyner_dur['Date'].dt.strftime('%m-%d'))
        # st.plotly_chart(last_three_hyner_dur)

if selected_page == 'Compare Training':
    col1, col2 = st.columns(2)
    startdate = df['date'].min()
    enddate = df['date'].max()
    with col1:
        date1 = pd.to_datetime(st.date_input('Start Date', startdate))
    with col2:
        date2 = pd.to_datetime(st.date_input("End Date",enddate))

    compare_data = st.checkbox("Compare Data")


    if compare_data and selected_page == 'Compare Training':
        compare_df = get_data_between_dates(start_date=date1, end_date=date2, df=df)
        #compare_df =  assign_week_number(compare_df)
        selected_year = compare_df[['year']]
        compare_distance_df =  compare_df[(compare_df['type'] != 'Elliptical') & (compare_df['type'] != 'WeightTraining')& (compare_df['type'] != 'Sauna')]

        compare_distance = distance_calc(compare_df)
        

        compare_distance_bar = stacked_bar_chart(compare_distance,x='starting_week',y='distance',labels={'starting_week':'Date','distance':'Distance'},title=f'Weekly Duration Totals by Activity for  Selected Time Period',orientation='v')
        st.plotly_chart(compare_distance_bar)

        compare_run_df = compare_df[compare_df['type'] == 'Run']
        longest_runs = compare_run_df['distance'].nlargest(3)
        
        st.write("The 3 Longest Runs:")
        for i, run in enumerate(longest_runs, start=1):
            st.write(f"{i}. {round(run,2)} miles")


        compare_duration = stacked_bar_chart(compare_df,y='starting_week',x='total_elapsed_time',labels={'starting_week':'Date','total_elapsed_time':'Time in Mins.'},title='Weekly Duration Totals by Activity',orientation='h')
        compare_duration.update_yaxes(tickmode='array', tickvals=compare_df['starting_week'], ticktext=compare_df['starting_week'].dt.strftime('%m-%d'))
        compare_duration.update_xaxes(showgrid=True, gridcolor='DarkSlateBlue')
        compare_duration.update_layout(legend=dict(title='Activity'))

        st.plotly_chart(compare_duration)


        
if selected_page == 'All Time':
#calculate total distance per activity type per year
    total_distance_per_year = df[df['type'] != 'Elliptical']
    total_distance_per_year = total_distance_per_year.groupby(['type', 'year'])['distance'].sum().reset_index()
    
    st.title("Distance by Year and Activity")

    # Show the figure
    distance_by_type = stacked_bar_chart(total_distance_per_year,x='year',y='distance',labels={'distance':'Total Distance (Miles)','year':'Year'}, title='Annual Distance Summary by Activity',orientation='v')
    total_distance_per_year['year_str'] = total_distance_per_year['year'].astype(str)

    distance_by_type.update_xaxes(tickmode='array', tickvals=total_distance_per_year['year'], ticktext=total_distance_per_year['year_str'])

    st.plotly_chart(distance_by_type) 

    # total time per activity per year 
    total_time_per_year = df.groupby(['type', 'year'])['total_elapsed_time'].sum().reset_index()
    
    st.title("Toal Duration by Year and Activity")

    duration_by_type = stacked_bar_chart(total_time_per_year,x='year',y='total_elapsed_time',labels={'total_elapsed_time':'Total Time (Minutes)','year':'Year'}, title='Annual Duration Summary by Activity',orientation='v')
    total_time_per_year['year_str'] = total_time_per_year['year'].astype(str)

    duration_by_type.update_xaxes(tickmode='array', tickvals=total_time_per_year['year'], ticktext=total_time_per_year['year_str'])


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


if selected_page == 'Map':
    st.title("Running Map")
    st.components.v1.iframe("http://localhost:3000", width=1000, height=800)
  