#***********************************************************************************************
# Project : AI-Driven Temperature Analysis for Better Educational Environments in Tanzania
# Task    : Model Deployment and Prediction 
#***********************************************************************************************
#==============================================================================================
# 1. Import the required libraries
#==============================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import calendar
from datetime import datetime
import datetime
  
import pickle
import streamlit as st

import tensorflow as tf
import keras
from keras.models import load_model
#--------------------------------------------------------------
# From sklearn import required classes
#--------------------------------------------------------------
from sklearn import preprocessing  
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
#================================================================================================================

#----------------------------------------------------------------------------
# 2. Load the saved ML model & Variables
#----------------------------------------------------------------------------
model = load_model('LSTM_model.h5')   # trained ML model

X_scaler     = pickle.load(open('Scaler_for_Inputdata.pkl','rb'))  # MinMaxScaler for the input data 
Tair_scaler = pickle.load(open('Scaler_for_Prediction.pkl', 'rb')) # MinMaxScaler for predicted temperature

no_prev_timepoints = pickle.load(open('Timepoints_Past.pkl','rb'))  # no. of previous timepoints data used to predict temp at next timepoint
n_future_tps = pickle.load(open('Timepoints_Future.pkl', 'rb'))    # no.of future timepoints temperature will be predicted

#----------------------------------------------------------------------------
# 3. Creating a function for data transformation & forecasting
#----------------------------------------------------------------------------
def Forecasting(df_initial):

    # 1. Copy the required columns for model prediction
    #df_initial = df_initial[df_initial['school_name']=="Kijichi Primary School"]
    df_initial = df_initial.reset_index(drop=True)
    
    df_initial = df_initial.rename(columns={"Date/Time EAT": "Date"})
    df_initial['Date'] = pd.to_datetime(df_initial['Date'],infer_datetime_format=True,dayfirst=True)
    
    req_cols = ['Date',
            'Avg_indoor_temp(hourly)', 'temperature_2m', 'relative_humidity_2m', 
            'direct_normal_irradiance','shortwave_radiation', 'wind_speed_10m', 
            'cloud_cover', 'NDVI(monthly)', 'NDWI(monthly)']
    df_initial = df_initial[req_cols]
    
    # 2. Data distribution transformation to handle the skewed dataset
    df_log = df_initial[['direct_normal_irradiance', 'shortwave_radiation', 'NDVI(monthly)']]
    log_trf = FunctionTransformer(func=np.log1p)
    df_log = log_trf.fit_transform(df_log)
    
    df_initial['direct_normal_irradiance'] = df_log['direct_normal_irradiance']
    df_initial['shortwave_radiation'] = df_log['shortwave_radiation']
    df_initial['NDVI(monthly)'] = df_log['NDVI(monthly)']
 
    # 3. Dataset to compare Future predictions
    df_initial['Timepoint'] = df_initial.index
    df_future = df_initial[-n_future_tps:]     # Note: 1st copy the data for future time points
    df_initial = df_initial[:-n_future_tps]
    
    # Future Forecasting
    df = df_initial[['Avg_indoor_temp(hourly)','temperature_2m','relative_humidity_2m', 
                 'direct_normal_irradiance','shortwave_radiation', 'wind_speed_10m', 
                 'cloud_cover','NDVI(monthly)', 'NDWI(monthly)']].values
    
    input_for_future_pred = df[-no_prev_timepoints:]
    
    input_for_future_pred_scaled = X_scaler.transform(input_for_future_pred)
    
    future_X = input_for_future_pred_scaled.reshape(1,input_for_future_pred_scaled.shape[0],input_for_future_pred_scaled.shape[1])
    future_predict = model.predict(future_X)
    future_predict = Tair_scaler.inverse_transform(future_predict)
    # new dates

    def predict_dates(num_prediction):
        last_date = df_initial['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1, freq ='H').tolist()
        return prediction_dates

    forecast_dates = predict_dates(n_future_tps)
    forecast_dates = forecast_dates[1:]
    
    forecast_dates = np.array(forecast_dates)

    forecast = future_predict
    forecast = forecast.reshape((-1))    

    True_Future = df_future['Avg_indoor_temp(hourly)'].values
    
    Timepoint_model      = df_initial['Timepoint']
    Timepoint_future    = df_future['Timepoint']
    df = df_initial['Avg_indoor_temp(hourly)'].values
    
    #===========================================================================
    # Model Evaluation Metrics
    #===========================================================================
    R2_future = round(r2_score(True_Future,forecast),4)
    rmse_future = round(root_mean_squared_error(True_Future,forecast),3)
    mae_future = round(mean_absolute_error(True_Future,forecast),3)
    cvrmse_future = round(100*rmse_future/np.mean(True_Future),1)
    nmbe_future = round(100*mae_future/np.mean(True_Future),1)
    
    # Not returning R2_future, cvrmse_future, nmbe_future
    return df_initial, df_future, rmse_future, mae_future, df, True_Future, forecast, Timepoint_model, Timepoint_future, forecast_dates
    
#========================================================================================================
# Streamlit related
#========================================================================================================

# 1. Add logos
#---------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.image("omdena.png", width=200)
with col2:
    st.image("opendeved.png", width=220)

# 2. Titel & Goal 
#----------------------------------------
st.header('Project: :blue[AI-Driven Temperature Analysis for Better Educational Environments in Tanzania]')
st.header("Prediction of indoor classroom temperature:thermometer:", divider='rainbow')

# 3. Take input data file (*.csv) from user
#---------------------------------------------------------------------------------
st.write("Please upload previous 10 days data in *.csv format: ")

inputdata = st.file_uploader("upload file", type={"csv", "txt"})

#df_initial = pd.read_csv(inputdata)
if inputdata is not None:
    df_initial = pd.read_csv(inputdata)
    st.write(df_initial)  # display the data

#-------------------------------------------------------------------------------------------------------
if st.button('Predict',type="primary"):
    
    # 1. Forecasting 
    df_initial1, df_future1, rmse_, mae_, df_, True_Future_, forecast_, Timepoint_model_, Timepoint_future_, forecast_dates_ = Forecasting(df_initial)
    
    #st.subheader("Data used for ML model", divider='rainbow')
    #st.write(df_initial1)

    #st.subheader("Ground Truth data to compare with model predictions", divider='rainbow')
    #st.write(df_future1)   

    #===========================================================================
    st.subheader("Prediction Metrics:", divider='rainbow')
        
    st.write("Root Mean Squared Error :")  
    st.write(rmse_)
    
    st.write("Mean Absolute Error :")
    st.write(mae_)  
    
    #===========================================================================
    st.subheader("Prediction Graphs:", divider='rainbow')
    
    #st.write(df_date_)
    #st.write(forecast_dates_)
    
    Indoor_Temp_min = round(df_initial['Avg_indoor_temp(hourly)'].min(),0)
    Indoor_Temp_max = round(df_initial['Avg_indoor_temp(hourly)'].max(),0)

    mpl.rc('axes', labelsize=18)
    mpl.rc('xtick', labelsize=18)
    mpl.rc('ytick', labelsize=18)
    font      = {'family':'arial', 'style':'normal', 'size':18}
    axis_font = {'family':'arial', 'style':'normal', 'size':18}
    plt.rc('font', **font)

    # 1. Full dataset for DL Model & True_Future
    #---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.plot(Timepoint_model_,df_, label='Full dataset for DL Model')
    ax.plot(Timepoint_future_,True_Future_, label='True future')
    ax.set_title('Avg_indoor_temp(hourly)')
    ax.set_xlabel("Time point #", axis_font)
    ax.set_ylabel("Temperature [°C]", axis_font)
    ax.legend(loc='upper left')
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #ax.set_xticks(rotation=70)
    ax.grid()
    st.pyplot(fig)
    
    # 2. Full dataset for DL Model & True_Future
    #---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 10))
   
    ax.plot(forecast_dates_,True_Future_, label='True future')
    ax.plot(forecast_dates_,forecast_, label='Predicted future')
    ax.set_title('Avg_indoor_temp(hourly) : True future vs. Predicted future')
    ax.set_xlabel("Date/Time", axis_font)
    ax.set_ylabel("Temperature [°C]", axis_font)
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim(Indoor_Temp_min-1,Indoor_Temp_max+1)
    ax.legend(loc='upper left')
    ax.grid()
    st.pyplot(fig)    

    # 3. True future vs. Predicted future
    #---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 10))
 
    ax.plot(True_Future_, label='True future')
    ax.plot(forecast_, label='Predicted future')
    ax.set_title('Avg_indoor_temp(hourly): True future vs. Predicted future')
    ax.set_xlabel("Time point #", axis_font)
    ax.set_ylabel("Temperature [°C]", axis_font)
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim(Indoor_Temp_min-1,Indoor_Temp_max+1)
    ax.legend(loc='upper left')
    ax.grid() 
    st.pyplot(fig)
    