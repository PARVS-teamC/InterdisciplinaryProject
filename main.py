from libreria import utility
import pandas as pd
from water_leak_detection import AnomalyDetector
import os


def read_leak_id(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return (file.read().strip())
    else:
        return "P01"

def write_leak_id(file_path, leak_id):
    with open(file_path, 'w') as file:
        file.write(str(leak_id))

def update_id(s):
    prefix = ''.join([char for char in s if char.isalpha()])
    number = ''.join([char for char in s if char.isdigit()])    
    incremented_number = int(number) + 1    
    num_zeros = len(number) - len(str(incremented_number))
    new_number = '0' * num_zeros + str(incremented_number)
    incremented_id = prefix + new_number    
    return incremented_id

def recupera_data(): #da Server
    a = utility()
    df_pressure = a.get_all_pressure()
    df_flow = a.get_all_flow()
    df = pd.merge(df_pressure, df_flow, on='timestamp')
    df = df.dropna()
    collected_data = df.set_index("timestamp")
    zone="DMA1"
    result={
            "zone": zone,
            "data":collected_data
        }
    return result

def recupera_info(): #da Server
    state= False #da db
    cnt_anomaly=0 #da db
    cnt_normal=0 #da db
    return state,cnt_anomaly,cnt_normal 

	
def recupera_periodo_anomalo(start_leak): #da Server
    #richiesta al server dei dati compresi tra start_leak-14 e start_leak
    a = utility()
    df_pressure = a.get_pressure_range(start_leak,14)
    df_flow = a.get_flow_range(start_leak,14)
    df = pd.merge(df_pressure, df_flow, on='timestamp')
    df = df.dropna()
    anomaly_data = df.set_index("timestamp")    
    
    return anomaly_data

def leak_detection():
    info= recupera_data()
    sensors_data = info["data"]
    dma=info["zone"]
    detector = AnomalyDetector()
    detector.split_data(sensors_data)    
    detector.train_()    
    detector.test_()    
    anomalies = detector.detect_anomalies()        
    state, count_anomaly, count_normal = recupera_info()
    result = detector.check_leakage(anomalies, state, count_anomaly, count_normal) 
    dict = {
        "DMA": dma,
        "current_situation": result["current_situation"],
        "count_anomaly": result["count_anomaly"],
        "count_normal": result["count_normal"]
    }
    
    if result["start_date_leak"] is not None: 
        anomaly_period=recupera_periodo_anomalo(result["start_date_leak"])
        culprit_sensor = detector.localize_leak(anomaly_period)
        leak_id = read_leak_id("Results/leaks")
        dict["leak_id"]=leak_id
        dict["localization"] = culprit_sensor
        dict["start_date_leak"] = result["start_date_leak"]
        write_leak_id("Results/leaks", update_id(leak_id))
    
    if result["stop_date_leak"] is not None:
        dict["stop_date_leak"] = result["stop_date_leak"]
    
    return dict

print(leak_detection())



