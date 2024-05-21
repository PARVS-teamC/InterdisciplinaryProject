# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:50:01 2024

@author: mirip
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def clean_(df):
    df['bt'] = df['bt'].str.replace('$numberInt:', '')
    df['bt']=df['bt'].astype(int)
    df['v'] = df['v'].str.replace('$numberDouble:', '')
    df['v']=df['v'].str.replace('$numberInt:', '')
    df['v']=df['v'].astype(float)
    print(df['v'])
    return df

def gen_data(start_time, duration_weeks, mean, std, start_id,tipo, arduino):
    #tipo: f flusso, p pressione
    end_time=start_time + timedelta(weeks=duration_weeks)
    #T= minuti, PER CAMBIARE FREQUENZA SI POUO FARE DA QUI
    date_range= pd.date_range(start=start_time,end=end_time,freq='5min')
    values=np.random.normal(loc=mean, scale=std,size=len(date_range))
    sensor_ids = [f'Virtual_ID{i}' for i in range(start_id, start_id + len(date_range))]
    unix_timestamps = date_range.astype(np.int64) // 10**9
    if tipo=='p':
        um='bar'
        type_= 'pressure'
    else: 
        um= 'l/s'
        type_='flow'
    data = pd.DataFrame({
       "_id": [f"$oid:virtual{i}" for i in range(start_id, start_id + len(date_range))],
       "bt": unix_timestamps,
       "u": [um] * len(date_range),
       "n": [type_] * len(date_range),
       "v": values,
       "bn": [arduino] * len(date_range)
     })
    return data, start_id + len(date_range)

    
    
    

pressure1_noleak = pd.read_csv("C:/Users/mirip/Downloads/Arduino_00_NO_leak_pressure_data.csv")
pressure2_noleak=pd.read_csv("C:/Users/mirip/Downloads/Arduino_01_NO_leak_pressure_data.csv")
pressure1_leak=pd.read_csv("C:/Users/mirip/Downloads/Arduino_00_YES_leak_pressure_data.bson.csv")
pressure2_leak= pd.read_csv( "C:/Users/mirip/Downloads/Arduino_01_YES_leak_pressure_data.bson.csv")

flow1_noleak = pd.read_csv("C:/Users/mirip/Downloads/Arduino00_NO_leak_flow_data.csv")
flow2_noleak=pd.read_csv("C:/Users/mirip/Downloads/Arduino_01_NO_leak_flow_data.csv")
flow1_leak=pd.read_csv("C:/Users/mirip/Downloads/Arduino00_YES_leak_flow_data.csv")
flow2_leak= pd.read_csv( "C:/Users/mirip/Downloads/Arduino_01_YES_leak_flow_data.csv")

#clean
pressure1_noleak= clean_(pressure1_noleak)
pressure2_noleak=clean_(pressure2_noleak)
pressure1_leak=clean_(pressure1_leak)
pressure2_leak=clean_(pressure2_leak)

flow1_noleak= clean_(flow1_noleak)
flow2_noleak=clean_(flow2_noleak)
flow1_leak=clean_(flow1_leak)
flow2_leak=clean_(flow2_leak)

#mean and std
#pressure
p_noleak1_avg=pressure1_noleak['v'].mean()
p_noleak1_std=pressure1_noleak['v'].std()

p_noleak2_avg=pressure2_noleak['v'].mean()
p_noleak2_std=pressure2_noleak['v'].std()

p_leak1_avg=pressure1_leak['v'].mean()
p_leak1_std=pressure1_leak['v'].std()

p_leak2_avg=pressure2_leak['v'].mean()
p_leak2_std=pressure2_noleak['v'].std()

#flow
f_noleak1_avg=flow1_noleak['v'].mean()
f_noleak1_std=flow1_noleak['v'].std()

f_noleak2_avg=flow2_noleak['v'].mean()
f_noleak2_std=flow2_noleak['v'].std()

f_leak1_avg=flow1_leak['v'].mean()
f_leak1_std=flow1_leak['v'].std()

f_leak2_avg=flow2_leak['v'].mean()
f_leak2_std=flow2_noleak['v'].std()


#scegli una data inziale
start_time=datetime.now()
start_id=0
#crea la tua query, si possono alternare periodi di leak e no leak
#a piacimento, quetso è per il caso autoencoder
pressure1_noleak, start_id= gen_data(start_time, 2, p_noleak1_avg, p_noleak1_std,start_id,'p','arduino_00')
pressure2_noleak, start_id = gen_data(start_time, 2, p_noleak2_avg, p_noleak2_std,start_id,'p','arduino_01')

flow1_noleak, start_id = gen_data(start_time, 2, f_noleak1_avg, f_noleak1_std,start_id,'f','arduino_00')
flow2_noleak, start_id = gen_data(start_time, 2, f_noleak2_avg, f_noleak2_std, start_id,'f','arduino_01')

start_time_leak = start_time + timedelta(weeks=2)

pressure1_leak,start_id = gen_data(start_time_leak, 3, p_leak1_avg, p_leak1_std,start_id,'p','arduino_00')
pressure2_leak, start_id = gen_data(start_time_leak, 3, p_leak2_avg, p_leak2_std, start_id,'p','arduino_01')

flow1_leak, start_id = gen_data(start_time_leak, 3, f_leak1_avg, f_leak1_std, start_id,'f','arduino_00')
flow2_leak, start_id = gen_data(start_time_leak, 3, f_leak2_avg, f_leak2_std, start_id,'f','arduino_00')

all_data_p = pd.concat([pressure1_noleak, pressure2_noleak, pressure1_leak, pressure2_leak])
all_data_f=  pd.concat([flow1_noleak, flow2_noleak, flow1_leak, flow2_leak])
# Save 
all_data_p.to_csv('C:/Users/mirip/Desktop/interdisciplinary/InterdisciplinaryProject/combined_sensor_data_p.csv', index=False)
all_data_f.to_csv('C:/Users/mirip/Desktop/interdisciplinary/InterdisciplinaryProject/combined_sensor_data_f.csv', index=False)
