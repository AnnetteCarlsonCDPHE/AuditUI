# -*- coding: utf-8 -*-
"""
try to just write defs here then call them in the 'spodaudit_main' script
"""

import pytz
import pandas as pd
import os
import re 
import numpy as np 
from io import BytesIO
import matplotlib.pyplot as plt
import streamlit as st 
from datetime import datetime, timedelta


def convert_uct2mst(df, utc_column, mst_column, time_only_column):
    # Convert the time column from string to datetime format
    df[utc_column] = pd.to_datetime(df[utc_column])

    # Convert UTC to MST
    utc_timezone = pytz.timezone('UTC')
    mst_timezone = pytz.timezone('US/Mountain')  # Mountain Standard Time
    df[mst_column] = df[utc_column].dt.tz_localize(utc_timezone).dt.tz_convert(mst_timezone)
    
    # Extract the time part (HH:MM:SS) and add as a new column
    df[time_only_column] = df[mst_column].dt.strftime('%H:%M')

    return df

def is_mdt(date):
    """Return True if the date is within the MDT period, False if within the MST period."""
    year = date.year
    second_sunday_march = datetime(year, 3, 8) + timedelta(days=(6 - datetime(year, 3, 8).weekday()))
    first_sunday_november = datetime(year, 11, 1) + timedelta(days=(6 - datetime(year, 11, 1).weekday()))
    return second_sunday_march <= date < first_sunday_november

def convert_mdt2mst(df):

    # Check if the index is timezone-naive and determine the appropriate timezone
    if df.index.tzinfo is None:
        sample_date = df.index[0]
        if is_mdt(sample_date):
            print("Localizing index to MDT.")
            df.index = df.index.tz_localize('US/Mountain', ambiguous='NaT', nonexistent='NaT')
        else:
            print("Localizing index to MST.")
            df.index = df.index.tz_localize('Etc/GMT+7', ambiguous='NaT', nonexistent='NaT')

    # Define the MDT and MST timezones
    mdt = pytz.timezone('US/Mountain')
    mst = pytz.timezone('Etc/GMT+7')

    # Check if the index is already localized to MST
    if df.index.tz == mst:
        print("The DataFrame is already in MST.")
    else:
        # Convert the index to MST if it's currently in MDT
        if df.index.tz == mdt:
            df.index = df.index.tz_convert(mst)
            print("Converted index from MDT to MST.")
        else:
            print("The DataFrame is already in a different timezone.")
            
    # Extract the time part (HH:MM:SS) into a new column
    df['MST_time_only'] = df.index.strftime('%H:%M')

    return df

def format_kestrel(df, keywords, text): #, units_dict
    # Filter columns based on keywords
    filtered_columns = [col for col in df.columns if any(re.search(keyword, col, re.IGNORECASE) for keyword in keywords)]
    #print(f"Filtered columns: {filtered_columns}")
    # Select only the filtered columns
    filtered_df = df[filtered_columns]

    # Rename the columns by adding the specified text
    new_column_names = {col: f"{text}{col}" for col in filtered_columns}
    formatted_df = filtered_df.rename(columns=new_column_names)
    
    return formatted_df

def average_kestrel(df):
    
    #convert everything to numbers for math
    df['Kestrel Temperature'] = pd.to_numeric(df['Kestrel Temperature'], errors='coerce')
    df['Kestrel Relative Humidity'] = pd.to_numeric(df['Kestrel Relative Humidity'], errors='coerce')
    df['Kestrel Station Pressure'] = pd.to_numeric(df['Kestrel Station Pressure'], errors='coerce')
    df['Kestrel Wind Speed'] = pd.to_numeric(df['Kestrel Wind Speed'], errors='coerce')
    df['Kestrel Compass Magnetic Direction'] = pd.to_numeric(df['Kestrel Compass Magnetic Direction'], errors='coerce')

    #st.write(df.dtypes)
    #convert wind dir to radians
    df['Polar WD'] = np.deg2rad(df['Kestrel Compass Magnetic Direction'])    
    #break ws into components to average
    df['U'] = df['Kestrel Wind Speed'] * np.sin(np.radians(df['Polar WD']))
    df['V'] = df['Kestrel Wind Speed'] * np.cos(np.radians(df['Polar WD']))
     
    
    #assign datetime index before resample
    df['kestrel datetime'] = pd.to_datetime(df['Kestrel FORMATTED DATE_TIME'])
    df = df.set_index('kestrel datetime')

    #1 minute average
    numeric_df = df.select_dtypes(include=['number'])
    df_resampled = numeric_df.groupby(pd.Grouper(freq='1T')).mean()
    
    # Reconstruct wind speed and direction from averaged U and V components
    df_resampled['kestrel_wind_speed_resampled'] = np.sqrt(df_resampled['U']**2 + df_resampled['V']**2)
    cor_angle_rad = (np.degrees(np.arctan2(df_resampled['U'], df_resampled['V'])) + 360) % 360
    df_resampled['kestrel_wind_direction_resampled']  = np.rad2deg(cor_angle_rad) % 360
    
    #add a time only column at the end for future
    # Extract the time part (HH:MM:SS) and add as a new column
    df_resampled['time_only'] = df_resampled.index.strftime('%H:%M')
    
    ## Drop multiple columns
    columns_to_drop = ['Kestrel Wind Speed', 'Kestrel Compass Magnetic Direction','U','V','Polar WD']
    df_resampled = df_resampled.drop(columns_to_drop, axis=1)
    
    #add units back in
    unit_dict = {
    'Kestrel Temperature': 'Kestrel Temperature (\u00b0 C)',
    'Kestrel Relative Humidity': 'Kestrel Relative Humidity (%)',
    'Kestrel Station Pressure': 'Kestrel Station Pressure (hPa)',
    'kestrel_wind_speed_resampled':'Kestrel Wind Speed Resampled (m/s)',
    'kestrel_wind_direction_resampled':'Kestrel Wind Direction Resampled (deg)'
}


    df_resampled = df_resampled.rename(columns=unit_dict)

    
    return df_resampled


#rename spod headers to be smaller/easier to deal with - not really needed but makes my brain happier 
def rename_columns_spod(df):
    # Define a dictionary where keys are patterns to search for and values are the new column names
    rename_dict = {
        r'.*_HUMID %': 'SPOD humidity (%)',
        r'.*_MV RAW': 'SPOD MVraw',
        r'.*_TVOC \(ppm\)': 'SPOD TVOC (ppm)',
        r'.*_PRESSURE MBAR': 'SPOD Pressure (hPa)',
        r'.*_TEMP C': 'SPOD Temperature (\u00b0 C)',
        r'.*_WIND DIRECTION': 'SPOD Wind Direction (deg)',
        r'.*_WIND SPEED': 'SPOD Wind Speed (m/s)'
    }

    # Iterate over the columns and apply the renaming based on patterns
    new_column_names = {}
    for col in df.columns:
        new_name = col  # Default to the original name
        for pattern, new_col_name in rename_dict.items():
            if re.search(pattern, col):
                new_name = new_col_name
                break
        new_column_names[col] = new_name

    # Rename the columns
    df.rename(columns=new_column_names, inplace=True)
    return df


def mergeKestrel(uploaded_files, header_row = 3):
    #init empty list to store DataFrames
    dfs = []
       
    
    # Read each file and append its DataFrame to the list
    for file in uploaded_files:
        #st.write(f"Reading file: {file.name}")  # Debugging print statement
        df = pd.read_csv(file, header=header_row)  # Read file from BytesIO
        dfs.append(df)
        
    # Check if we have DataFrames to concatenate
    if dfs:
        kestrel_merged = pd.concat(dfs, ignore_index=True)
    else:
        kestrel_merged = pd.DataFrame()  # Return an empty DataFrame if no files are uploaded
    
    return kestrel_merged

#emulating Leah's audit analysis code (zero_air_analysis)

def shorten_to_analysis(df,time_column, start_time,end_time):
    df = df.set_index(time_column)
    analysis_data = df[(df.index >= start_time) & (df.index <= end_time)]
    return analysis_data


def shorten_to_analysis_kestrel(df,time_column, start_time,end_time):
    
    df = df.set_index(time_column)
    analysis_data = df[(df.index >= start_time) & (df.index <= end_time)]
    return analysis_data

#zero air audit
def create_plotszero(df,x_keyword,y_keyword,start_time,end_time,time_column):
   # Find columns based on keywords
    x_column = next((col for col in df.columns if x_keyword in col), None)
    y_column = next((col for col in df.columns if y_keyword in col), None)
    
    #convert start/end times from strings to dt before if loop 
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    # Check if both columns are found
    if x_column and y_column and time_column in df.columns:
        
        df[time_column] = pd.to_datetime(df[time_column])
        # Filter data by the specified time range (via buttons in UI)
        filtered_df = shorten_to_analysis(df, time_column, start_time, end_time)
        
        #ID 20 min before audit
        starttime_20minbefore = start_time - pd.Timedelta(minutes=20)
        pre_filtered_df = shorten_to_analysis(df, time_column, starttime_20minbefore, start_time)
        
        if not filtered_df.empty and not pre_filtered_df.empty:
            fig, ax = plt.subplots(figsize=(10, 10))
            
            ax.scatter(pre_filtered_df[x_column], pre_filtered_df[y_column], color='red', label='Ambient air (20 min)')
            
            ax.scatter(filtered_df[x_column], filtered_df[y_column], color='blue', label='Zero air period')

            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.legend()
            plt.gcf().autofmt_xdate()
            st.pyplot(fig)
        else:
            st.write(f"No data found in the specified time range: {start_time} to {end_time}.")
    else:
        st.write(f"Columns with keywords '{x_keyword}' and/or '{y_keyword}' and/or '{time_column}' not found in the DataFrame.")
        
        
#cal gas audit:

def create_plots(df,x_keyword,y_keyword,start_time,end_time,time_column):
   # Find columns based on keywords
    x_column = next((col for col in df.columns if x_keyword in col), None)
    y_column = next((col for col in df.columns if y_keyword in col), None)
    
    # Check if both columns are found
    if x_column and y_column and time_column in df.columns:
        # Filter data by the specified time range (via buttons in UI)
        filtered_df = shorten_to_analysis(df, time_column, start_time, end_time)
        
        
        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(10, 10))
            
            ax.scatter(filtered_df[x_column], filtered_df[y_column])
            
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
           
            plt.gcf().autofmt_xdate()
            st.pyplot(fig)
        else:
            st.write(f"No data found in the specified time range: {start_time} to {end_time}.")
    else:
        st.write(f"Columns with keywords '{x_keyword}' and/or '{y_keyword}' and/or '{time_column}' not found in the DataFrame.")
        
        
def create_plotscal(df, x_keyword, y_keyword, keyword, keyword2, start_time,end_time,time_column, slope=None, intercept=None, cal_column_keyword=None):
   # Find columns based on keywords
    x_column = next((col for col in df.columns if x_keyword in col), None)
    y_column = next((col for col in df.columns if y_keyword in col), None)
    cal_column = next((col for col in df.columns if cal_column_keyword in col), None)
    
    # Check if both columns are found
    if x_column and y_column and time_column in df.columns:
        # Filter data by the specified time range (via buttons in UI)
        filtered_df = shorten_to_analysis(df, time_column, start_time, end_time)
        #calc rate of change + find where data level off to <1mv change  
        dMV_col = next((col for col in filtered_df.columns if keyword in col), None)
        
        if dMV_col is None:
            raise ValueError(f"No column found with keyword '{keyword}'")
            
        filtered_df['rate of change'] = filtered_df[dMV_col].diff().abs()
        stable_df = filtered_df[filtered_df['rate of change'] <= 1].dropna()
        
        #make % difference calcuation 
        tvoc_col = next((col for col in stable_df.columns if keyword2 in col), None)
        if tvoc_col is None:
            raise ValueError(f"No column found with keyword '{keyword}'")
        
        # Ensure the tvoc column is numeric
        stable_df[tvoc_col] = pd.to_numeric(stable_df[tvoc_col], errors='coerce')
        
        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot the calibration curve if slope and intercept are provided
            if slope is not None and intercept is not None:
                expected_y = slope * filtered_df[cal_column] + intercept
                ax.plot(filtered_df[x_column], expected_y, color='green', label='Calibration Curve (MV = m(TVOC)+b)')
            
            
            ax.scatter(filtered_df[x_column], filtered_df[y_column], color='black',s=10 , label = 'Reported MV')
            ax.scatter(stable_df[x_column], stable_df[y_column], color='blue', s=40, marker='*',label = 'Data used in % difference computation')
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.legend()
            plt.gcf().autofmt_xdate()
            st.pyplot(fig)
        else:
            st.write(f"No data found in the specified time range: {start_time} to {end_time}.")
    else:
        st.write(f"Columns with keywords '{x_keyword}' and/or '{y_keyword}' and/or '{time_column}' not found in the DataFrame.")

def compute_basic_stats(df,start_time,end_time,time_column):
    """

    """
    filtered_df = shorten_to_analysis(df, time_column, start_time, end_time)

    stats = {}
    
    # Filter out columns of type datetime or timedelta
    numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

    stats['Minimum'] = filtered_df[numeric_columns].min().round(decimals=1)
    stats['Median'] = filtered_df[numeric_columns].median().round(decimals=1)
    stats['Maximum'] = filtered_df[numeric_columns].max().round(decimals=1)
    stats['Mean'] = filtered_df[numeric_columns].mean().round(decimals=1)
    stats['SD'] = filtered_df[numeric_columns].std().round(decimals=1)

    statsdf = pd.DataFrame(stats)

    return(statsdf)   
    
def percentdiff(df, keyword, start_time, end_time, time_column, keyword2, Gas_concentration): 
    #convert start/end times from strings to dt before if loop 
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Filter data by the specified time range (via buttons in UI)
    filtered_df = shorten_to_analysis(df, time_column, start_time, end_time)
    
    #make sure gas conc input is a float
    Gas_concentration = float(Gas_concentration)
    #calc rate of change + find where data level off to <1mv change  
    dMV_col = next((col for col in filtered_df.columns if keyword in col), None)
    
    if dMV_col is None:
        raise ValueError(f"No column found with keyword '{keyword}'")
        
    filtered_df['rate of change'] = filtered_df[dMV_col].diff().abs()
    stable_df = filtered_df[filtered_df['rate of change'] <= 1].dropna()
    
    st.dataframe(stable_df, use_container_width=True)
    
    #make % difference calcuation 
    tvoc_col = next((col for col in stable_df.columns if keyword2 in col), None)
    if tvoc_col is None:
        raise ValueError(f"No column found with keyword '{keyword}'")
    
    # Ensure the tvoc column is numeric
    stable_df[tvoc_col] = pd.to_numeric(stable_df[tvoc_col], errors='coerce')
    
    percent_diff_array = ((stable_df[tvoc_col] - Gas_concentration)/Gas_concentration)*100
    percent_diff_array.name = 'TVOC_%_Difference'
    
    #get one data point. 
    avg_percent_diff = {}
    avg_percent_diff['TVOC_%_Difference_avg'] = percent_diff_array.mean().round(decimals=1)
    avg_percent_diff['num_datapoints'] = len(stable_df)
    
  #  pd_df = avg_percent_diff.join(num_datapoints)
    avgdf = pd.DataFrame([avg_percent_diff])
 
    return avgdf


def plot_metdata(df1, time_columndf1, df2, time_columndf2, keywords, start_time, end_time):
    if len(keywords) != 5:
        raise ValueError("Exactly 5 keywords must be provided.")

    # Function to find columns that match a keyword using substring match
    def find_matching_columns(df, keywords):
        matching_columns = []
        for col in df.columns:
            for keyword in keywords:
                if re.search(keyword, col, re.IGNORECASE):  # Case insensitive match
                    matching_columns.append(col)
                    break  # Once a match is found, break out of inner loop
        return matching_columns

    # Convert start/end times from strings to datetime
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Convert time columns in df1 and df2 to datetime if not already
    df1[time_columndf1] = pd.to_datetime(df1[time_columndf1])
    df2[time_columndf2] = pd.to_datetime(df2[time_columndf2])

    # Filter data by the specified time range
    filtered_df1 = df1[(df1[time_columndf1] >= start_time) & (df1[time_columndf1] <= end_time)]
    filtered_df2 = df2[(df2[time_columndf2] >= start_time) & (df2[time_columndf2] <= end_time)]
    
    
    # Create subplots
    num_plots = len(keywords)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 5 * num_plots))
    fig.tight_layout(pad=5.0)

    for i, keyword in enumerate(keywords):
        # Find matching columns in filtered_df1 and filtered_df2
        matching_cols_df1 = find_matching_columns(filtered_df1, [keyword])
        matching_cols_df2 = find_matching_columns(filtered_df2, [keyword])

        if matching_cols_df1 and matching_cols_df2:
            for col1, col2 in zip(matching_cols_df1, matching_cols_df2):
                axes[i].plot(filtered_df1[time_columndf1], filtered_df1[col1], label=f'{col1}')
                axes[i].plot(filtered_df2[time_columndf2], filtered_df2[col2], label=f'{col2}', linestyle='--')
            axes[i].set_title(f'Time Series for {keyword}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(keyword)
            axes[i].legend()
        else:
            raise KeyError(f"No matching columns found for keyword '{keyword}' in both datasets.")

    # Show plots
    plt.gcf().autofmt_xdate()
    st.pyplot(fig)


    
    
    

    
    
    
    
    
    

    