# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:17:42 2024

@author: aacarlso
"""

import pandas as pd
import streamlit as st 
from datetime import datetime 
from spodaudit_functions import * 


# Store uploaded datasets in session state
if 'spod' not in st.session_state:
    st.session_state.spod = None
if 'kestrel' not in st.session_state:
    st.session_state.kestrel = pd.DataFrame()

# Formating UI page:
tabs = st.sidebar.radio("Navigation - select one:", ["Zero Air", "Cal Gas", "Meteorological Comparison"])

st.sidebar.title("Please enter start and end time (MST - 24hr format), hit return after each entry")
start_time = st.sidebar.text_input("Start Time (hh&#58;mm)")
end_time = st.sidebar.text_input("End Time (hh&#58;mm)")

if tabs == "Cal Gas":
    st.sidebar.title("Cal Gas Information")
    slope = st.sidebar.number_input('Enter the slope of the calibration curve here:')
    intercept = st.sidebar.number_input('Enter the intercept of the calibration curve here:')
    cal_column = st.sidebar.text_input('Enter the column keyword for the calibration variable here (ie: TVOC (only for now)):')
    Gas_concentration = st.sidebar.number_input('Enter the audit gas concentration here (number only)')

# Create sidebar for uploading datasets
st.sidebar.title("Upload Data as a .csv")
spod_file = st.sidebar.file_uploader("Upload SPOD Data",type=['csv'])
    
# Load SPOD data
if spod_file is not None:
    st.session_state.spod = pd.read_csv(spod_file, header=3)
    st.session_state.spod = convert_uct2mst(st.session_state.spod, 'Date', 'MST_datetime','Spod_time_only') #spod_time_only is in MST 
    st.session_state.spod = rename_columns_spod(st.session_state.spod)

if tabs == "Meteorological Comparison":
    st.markdown("there is an error for 'FORMATTED DATE_TIME' until you upload a file, it doesn't interfere with processing", unsafe_allow_html=True)
    # Upload multiple Kestrel files (test with just one I merged in a different script)
    kestrel_files = st.sidebar.file_uploader("Upload Kestrel Data", type=['csv'], accept_multiple_files=True) #, 
    kestrel_merged = mergeKestrel(kestrel_files, header_row = 3)
    #delete second row headers from each file
    kestrel_merged = kestrel_merged[kestrel_merged['FORMATTED DATE_TIME'] != 'yyyy-MM-dd hh:mm:ss a']
    st.session_state.kestrel_merged = kestrel_merged
    keyword_kestcol = ['FORMATTED','humidity','station pressure','temperature','Magnetic', 'wind speed']
    text = 'Kestrel '
    st.session_state.kestrel_merged = format_kestrel(st.session_state.kestrel_merged, keyword_kestcol, text)
    


# Define function for zero air (SPOD only)
def process_spod_zero(start_time, end_time):
    if st.session_state.spod is not None:
        #print header just to check SPOD upload
        st.write("Processed SPOD Data Header (all) - check", st.session_state.spod.head())
        
        #timeseries of voltage for just zero air audit 
        st.write('Timeseries for zero audit period')
        create_plotszero(st.session_state.spod,'MST_datetime','MV',start_time,end_time,'Spod_time_only')
        
        #stats for zero air audit
        st.write('SPOD statistics for the zero audit period:')      
        spod_stats2 = compute_basic_stats(st.session_state.spod,start_time,end_time,'Spod_time_only')
        st.dataframe(spod_stats2, use_container_width=True)
        
# Define function for processing cal gas (SPOD only)
def process_spod_cal(start_time, end_time, slope, intercept, cal_column):
    if st.session_state.spod is not None:
        #print header just to check SPOD upload
        st.write("Processed SPOD Data (all) Header - check", st.session_state.spod.head()) 
               
        #timeseries/stats of voltage for just cal gas part of audit 
        st.write('Timeseries for cal gas audit period')
        create_plotscal(st.session_state.spod,'MST_datetime','MV','MV', 'TVOC',start_time,end_time,'Spod_time_only', slope, intercept, cal_column) 
        
        st.write('SPOD statistics for the cal gas audit period:')      
        spod_stats2 = compute_basic_stats(st.session_state.spod,start_time,end_time,'Spod_time_only')
        st.dataframe(spod_stats2, use_container_width=True)
    
        # % difference: find where data change < 1mv then calc 1 value for % diff. 
        # just test that it gets the correct data 
        st.write('SPOD Percent Difference: Where data <1 mv change')
        spod_percentdiff = percentdiff(st.session_state.spod,'MV', start_time, end_time, 'Spod_time_only', 'TVOC', Gas_concentration)
        st.dataframe(spod_percentdiff)       
       
        # Add more processing as needed

# Define function for comparing the weather data (kestrel + SPOD)

def process_meteorologicaldata(start_time, end_time): 
    if st.session_state.spod is not None and st.session_state.kestrel is not None:
        #print header just to check SPOD upload
        st.write("Processed SPOD Data Header - check", st.session_state.spod.head())
        #average Kestrel data and print header to check it looks alright        
        st.session_state.kestrel_avg = average_kestrel(st.session_state.kestrel_merged)
        st.session_state.kestrel_avg = convert_mdt2mst(st.session_state.kestrel_avg)
        st.write("Averaged (1-minute) Kestrel Data Header - check", st.session_state.kestrel_avg.head())  
        #stats table output for both datasets - separately but use start/end time (display side by side)
        col1, col2 = st.columns(2)
        with col1:
            st.write('SPOD statistics for the audit period:')
            spod_stats2 = compute_basic_stats(st.session_state.spod,start_time,end_time,'Spod_time_only')
            st.dataframe(spod_stats2, use_container_width=True)
        
        with col2:
            st.write('Kestrel statistics for the audit period:')
            kestrel_stats = compute_basic_stats(st.session_state.kestrel_avg,start_time,end_time,'MST_time_only')
            st.dataframe(kestrel_stats, use_container_width=True)
        #write new plot code: time series of all vars in common
        st.write('Time series plots for meteorlogical variables')
        
        keywords3 = ['Temperature','Humidity','Pressure','Wind Speed','Wind Direction']
        time_columndf1='Spod_time_only'
        time_columndf2='MST_time_only'
        plot_metdata(st.session_state.spod, time_columndf1 , st.session_state.kestrel_avg,time_columndf2, keywords3, start_time, end_time)



# Main section to display content based on selected tab
    
if tabs == "Zero Air": 
    st.title("Zero Air")
    
    process_spod_zero(start_time, end_time)
    

elif tabs == "Cal Gas":
    st.title("Cal Gas")
    
    process_spod_cal(start_time, end_time, slope, intercept, cal_column)

    
elif tabs == "Meteorological Comparison":
    st.title("Meteorological Comparison")
    
    process_meteorologicaldata(start_time, end_time)
 