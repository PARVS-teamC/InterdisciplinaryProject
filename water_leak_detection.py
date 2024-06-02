import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras import layers
from matplotlib import pyplot as plt
from libreria import utility
import os
import time
import tensorflow as tf
import random

class AnomalyDetector:
    def __init__(self, window_size=8640, window_seg=288):
        self.window_size = window_size
        self.window_seg = window_seg

    def zscore(self,df, df_train):
        training_mean = df_train.mean()
        training_std = df_train.std()

        df_norm = (df - training_mean) / training_std
        df_norm = df_norm.dropna() 
        return df_norm 

    def segmentation(self,df):
        segments = []
        num_segments = len(df) - self.window_seg + 1
        for i in range(num_segments):
            segment = df.iloc[i:i+self.window_seg, :]
            segments.append(segment)
        return segments
    
    def get_index(self,df,col_name_list):
        index_list=[]
        for name in col_name_list:
            index = df.columns.get_loc(name)
            index_list.append(index)
        return index_list
    
    def split_data(self,df):
        self.df_train = df.iloc[:4033,] #two weeks of data
        self.df_test = df.iloc[4033:,]
    
    
    def check_daily_condition(self,day):
        num_columns = len(day.columns)
        half_columns = num_columns // 2
            
        sensor_conditions = []
        for column in day.columns:
            total_values = day[column].count()  # Total values for the sensor
            num_true = day[column].sum()        # Number of true values for the sensor
            sensor_conditions.append(num_true >= 0.8 * total_values)
        if sum(sensor_conditions) < half_columns:
            return False
        return True
    
    def preprocess_data(self,df):
        df_value = df.copy()
        for column in df.columns:
            df_value[column] = self.zscore(df[column],self.df_train[column])
        #Apply segmentation 
        x=self.segmentation(df_value)
        x=np.array(x)
        return x
    
    def build_model(self):
        self.x_train=self.preprocess_data(self.df_train)
        #Define the neural network
        model = keras.Sequential(
            [
                layers.Input(shape=(self.x_train.shape[1], self.x_train.shape[2])),
                layers.Conv1D(
                    filters=32,
                    kernel_size=7,
                    padding="same",
                    strides=1,
                    activation="tanh",
                ),
                layers.AveragePooling1D(pool_size=2),
                layers.Conv1D(
                    filters=16,
                    kernel_size=7,
                    padding="same",
                    strides=1,
                    activation="tanh",
                ),
                layers.AveragePooling1D(pool_size=2),
                layers.Conv1DTranspose(
                    filters=16,
                    kernel_size=7,
                    padding="same",
                    strides=4,
                    activation="tanh",
                ),
                layers.AveragePooling1D(pool_size=2),
                layers.Conv1DTranspose(
                    filters=32,
                    kernel_size=7,
                    padding="same",
                    strides=4,
                    activation="tanh",
                ),
                layers.AveragePooling1D(pool_size=2),
                #last layer
                layers.Flatten(),
                layers.Dense(288 * 4, activation='linear'),
                layers.Reshape((288, 4))
            ]
        )

        #Compile and fit
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        model.summary()
        history = model.fit(
            self.x_train,
            self.x_train,
            epochs=50,
            batch_size=144,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss")
            ],
        )
        return model
    
    def train_(self):
        #Predict on training and evaluate maximum average error
        self.model=self.build_model()
        self.x_train_pred = self.model.predict(self.x_train)

    def get_threshold(self):   
        train_mae_loss = np.mean(np.abs(self.x_train_pred - self.x_train), axis=1)
        max_train_mae=np.max( train_mae_loss)
        threshold=max_train_mae
        return threshold
    
    def test_(self):
        self.x_test=self.preprocess_data(self.df_test)
        self.x_test_pred = self.model.predict(self.x_test)
    
    def detect_anomalies(self):
        test_mae_loss = np.mean(np.abs(self.x_test_pred - self.x_test), axis=1)
        anomalies = test_mae_loss > self.get_threshold()
        anomalies_tot= np.any(anomalies, axis=1)

        anomaly_indices_per_column = {}
        for column in range(len(anomalies[0])):
            anomaly_indices_per_column[column] = []
            for data_idx in range(self.window_seg - 1, len(self.df_test) - self.window_seg + 1):
                if anomalies[data_idx, column]:
                    anomaly_indices_per_column[column].append(data_idx)

        anomalies_df = pd.DataFrame(anomalies, columns=self.df_test.columns)
        return anomalies_df
    
    def check_leakage(self,anomalies_df,current_situation,count_anomaly,count_normal):
        start_date=pd.Timestamp(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(self.df_test.index[0])))
        end_date=pd.Timestamp(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(self.df_test.index[-1])))
        window_size = pd.Timedelta(days=1)
        start_date_leak=None
        stop_date_leak=None
        current_date = start_date
        days_without_leakage=count_normal
        days_with_leakage=count_anomaly

        while current_date <= end_date:
            window_start_slot = (current_date - start_date) // pd.Timedelta(minutes=5)
            window_end_slot= (current_date - start_date + window_size) // pd.Timedelta(minutes=5)
            window = anomalies_df[(anomalies_df.index >= window_start_slot) & (anomalies_df.index <= window_end_slot)]
            window = window.astype(bool)  # ensure the window contains only boolean values

            if self.check_daily_condition(window):
                days_with_leakage+=1
                days_without_leakage=0
            else:
                days_without_leakage+=1
                days_with_leakage=0
            
            if days_with_leakage>=14 and current_situation==False:
                #localization
                current_situation=True        
                days_without_leakage=0
                start_date_leak=current_date
            elif days_without_leakage>=14 and current_situation==True:
                current_situation=False
                days_with_leakage=0
                stop_date_leak=current_date
            
            current_date += pd.Timedelta(days=1)
        
        result={
            "current_situation": current_situation,
            "count_anomaly": days_with_leakage,
            "count_normal": days_without_leakage,
            "start_date_leak": start_date_leak,
            "stop_date_leak": stop_date_leak
        }
        return result
    
    def localize_leak(self,anomaly_period):
        correlation_normal = self.df_train.corr()
        correlation_anomaly = anomaly_period.corr()
        correlation_diff = (correlation_normal - correlation_anomaly).abs()
        correlation_change_score = correlation_diff.sum(axis=1)
        culprit_sensor = correlation_change_score.idxmax()
        return culprit_sensor





