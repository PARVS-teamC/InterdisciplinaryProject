from libreria import utility
import pandas as pd
from water_leak_detection import AnomalyDetector

class Test():
    def __init__(self):
        self.conn= utility()
        self.detector= AnomalyDetector('./')
        self.leak_detection_forall()
        
            
    def getUsersList(self):
        users = self.conn.get_users_list()
        return users
    
    def getTestData(self):
        pressure_data=self.conn.get_pressure_test(self.sensors)
        flow_data=self.conn.get_flow_test(self.sensors)
        data=pd.merge(pressure_data, flow_data, on='timestamp')
        data=data.dropna()
        test_data=data.set_index("timestamp")
        return test_data
    
    def getAnomalyPeriod(self,start_leak): 
        pressure_anomaly = self.conn.get_pressure_range(start_leak,14)
        flow_anomaly= self.conn.get_flow_range(start_leak,14)
        data = pd.merge(pressure_anomaly,flow_anomaly, on='timestamp')
        data = data.dropna()
        anomaly_data = data.set_index("timestamp")            
        return anomaly_data
   

    def leak_detection(self):
        test_data= self.getTestData()
        self.detector.test_(test_data)         
        anomalies = self.detect_anomalies()
        state_info=self.conn.get_state_values(self.dma)   
        result = detector.check_leakage(test_data,anomalies,state_info) 
        dict = {
            "DMA": self.dma,
            "current_situation": result["current_situation"],
            "count_anomaly": result["count_anomaly"],
            "count_normal": result["count_normal"]
        }
        
        if result["start_date_leak"] is not None:         
            anomaly_period=self.getAnomalyPeriod(result["start_date_leak"])
            culprit_sensor = detector.localize_leak(test_data,anomaly_period)
            dict["localization"] = culprit_sensor
            dict["start_date_leak"] = result["start_date_leak"]
            '''
            leak_id = read_leak_id("Results/leaks") #Stefano?
            dict["leak_id"]=leak_id
            write_leak_id("Results/leaks", update_id(leak_id)) #Stefano?'''
        
        if result["stop_date_leak"] is not None:
            dict["stop_date_leak"] = result["stop_date_leak"]
        return dict
    

    def leak_detection_forall(self):
        users_list = self.getUsersList()
        for user in users_list:
            info=self.conn.get_sensors_info(user)
            if info is not None:
                for zone in info:
                    self.dma = zone['zone']
                    self.sensors = zone['sensors']
                    self.leak_detection()
                    
                    
    
test_obj=Test()          