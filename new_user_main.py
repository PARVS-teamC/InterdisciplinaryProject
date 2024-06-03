from libreria import utility
import pandas as pd
from water_leak_detection import AnomalyDetector

class Training():
    def __init__(self):
        self.conn= utility()
        self.detector= AnomalyDetector('./')
        self.getThresholdforall()
            
    def getUsersList(self):
        users = self.conn.get_users_list()
        return users
    
    def getTrainData(self):
        pressure_data=self.conn.get_pressure_per_sensors(self.sensors)
        flow_data=self.conn.get_flow_per_sensors(self.sensors)
        data=pd.merge(pressure_data, flow_data, on='timestamp')
        data=data.dropna()
        train_data=data.set_index("timestamp")
        return train_data

    def getThreshold(self):
        train_data=self.getTrainData()       
        self.detector.train_(train_data)
        threshold=self.detector.get_threshold()
        dict = {
            "DMA": self.dma,
            "threshold": threshold,
            "std": self.detector.std_list,
            "mean": self.detector.mean_list
        }
        return dict

    def getThresholdforall(self):
        users_list = self.getUsersList()
        for user in users_list:
            info=self.conn.get_sensors_info(user)
            if info is not None:
                for zone in info:
                    self.dma = zone['zone']
                    self.sensors = zone['sensors']
                    self.getThreshold()
    
train_obj=Training()                
