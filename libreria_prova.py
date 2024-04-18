from pymongo import MongoClient
from datetime import datetime, timedelta

class utility():

    def __init__(self):
      try:
          self.client = MongoClient('mongodb+srv://admin_teamC:yGkHcCGezKMa1VET@waterleakage.zkwupj6.mongodb.net/?retryWrites=true&w=majority')
          self.db = self.client["Data"]
          self.user_collection = self.db["user_data"]
          self.pressure_collection = self.db["pressure_data"]
          self.flow_collection = self.db["flow_data"]

      except Exception as e:
            message = "An error occurred while running the service:PROVA_MSG" + str(e)
            raise Exception(message)

    def retrieve_userslist(self):
        users_list=[]
        project={
                'user_id': 1,
                'full_name': {
                    '$concat': [
                        '$name', ' ', '$surname'
                    ]
                },
                '_id': 0
                }

        result = self.user_collection.find(filter={},projection=project)
        for user in result:
            users_list.append(user)
        return users_list

    def retrieve_usersinfo(self,user_ids):
        users_info=[]
        for id in user_ids:
            filter = {"user_id": id}
            info = self.user_collection.find_one(filter, {"_id": 0, "user_id":1, "password": 1, "address": 1, "sensors.sensor_id": 1})
            users_info.append(info)
        return users_info

    def retrieve_password(self,user_ids):
        user_pwd=[]
        for id in user_ids:
            filter = {"user_id": id}
            pwd = self.user_collection.find_one(filter, {"_id": 0, "user_id":1, "password": 1})
            user_pwd.append(pwd)
        return user_pwd

    def retrieve_usersensors(self,user_ids):
        user_sensors=[]
        for id in user_ids:
            filter = {"user_id": id}
            info = self.user_collection.find_one(filter, {"_id": 0, "sensors.sensor_id": 1})
            sensor_ids = [sensor["sensor_id"] for sensor in info.get("sensors", [])]
            user_dict = {"user_id": id, "sensor_ids": sensor_ids}
            user_sensors.append(user_dict)
        return user_sensors

    def retrieve_sensorsinfo(self,sensor_ids):
        sensors_info=[]
        for sensor_id in sensor_ids:
            filter = {"sensors.sensor_id": sensor_id}
            projection = {
                "_id": 0,
                "sensors.$": 1
            }
            info = self.user_collection.find_one(filter, projection)

            if  info:
                sensor_data = info.get("sensors")[0]
                location = sensor_data.get("location")
                dma = sensor_data.get("dma")
                sensor_type = sensor_data.get("sensor_type")


                sensor_dict = {
                    "sensor_id": sensor_id,
                    "sensor_type": sensor_type,
                    "location": location,
                    "dma": dma
                }

                sensors_info.append(sensor_dict)
        return sensors_info

    def retrieve_flow_data(self,starting_time_unix, duration_days, sensor_ids):
        starting_time_dt = datetime.utcfromtimestamp(starting_time_unix)
        end_time_dt = starting_time_dt + timedelta(days=duration_days)

        query = {
            "$and": [
                {"bt": {"$gte": starting_time_dt}},
                {"bt": {"$lt": end_time_dt}},
                {"bn": {"$in": sensor_ids}}
            ]
        }

        sensors_info = []
        result = self.flow_collection.find(query, {"bt": 1, "v": 1, "bn": 1})

        for document in result:
            sensor_id = document.get("bn")
            timestamp = document.get("bt")  
            value = document.get("v")

            sensor_dict = {
                "sensor_id": sensor_id,
                "timestamp": timestamp,
                "value": value
            }
            sensors_info.append(sensor_dict)
        return sensors_info   
       

    def retrieve_all_flow_data(self,sensor_id):
        sensor_data = []
        query = {"bn": sensor_id}
        result = self.flow_collection.find(query, {"bt": 1, "v": 1, "bn": 1})
        for document in result:
            timestamp = document.get("bt")
            value = document.get("v")
            sensor_dict = {
                # "sensor_id": sensor_id,
                "timestamp": timestamp,
                "value": value
            }
            sensor_data.append(sensor_dict)
        return sensor_data
     
    def retrieve_pressure_data(self, starting_time_unix, duration_days, sensor_ids):
        starting_time_dt = datetime.utcfromtimestamp(starting_time_unix)
        end_time_dt = starting_time_dt + timedelta(days=duration_days)
      
        query = {
            "$and": [
                {"bt": {"$gte": starting_time_dt}},
                {"bt": {"$lt": end_time_dt}},
                {"bn": {"$in": sensor_ids}}
            ]
        }
        
        result = self.pressure_collection.find(query, {"bt": 1, "v": 1, "bn": 1})
        sensors_info = []
        for document in result:
            sensor_id = document.get("bn")  
            timestamp = document.get("bt")
            value = document.get("v")

            sensor_dict = {
                "sensor_id": sensor_id,
                "timestamp": timestamp,
                "value": value
            }

            sensors_info.append(sensor_dict)
        return sensors_info
    
    def retrieve_all_pressure_data(self,sensor_id):
        sensor_data = []
        query = {"bn": sensor_id}
        result = self.pressure_collection.find(query, {"bt": 1, "v": 1, "bn": 1})
        for document in result:
            timestamp = document.get("bt")
            value = document.get("v")
            sensor_dict = {
                # "sensor_id": sensor_id,
                "timestamp": timestamp,
                "value": value
            }
            sensor_data.append(sensor_dict)
        return sensor_data

#main di test
if __name__ == "__main__":
    a=utility()
    #---------------------USER DATA --------------------------------------------
    # user_ids = [user['user_id'] for user in a.retrieve_userslist()]
    # print("Users:", user_ids)
    # user_info=a.retrieve_usersinfo(user_ids)
    # print("Users info:", user_info)
    # pwd_user= a.retrieve_password(user_ids)
    # print("Passwords:", pwd_user)
    # sensors_list=a.retrieve_usersensors(user_ids)
    # sensor_ids = [user['sensor_ids'] for user in sensors_list]
    # print("Sensors:", sensor_ids)
    # sensors_info=a.retrieve_sensorsinfo(sensor_ids[0])
    # print("Info for sensor1,sensor2,sensor3:",sensors_info)
    #---------------------PRESSURE DATA ----------------------------------------
    sensor_ids= ['sensor_6', 'sensor_4']
    pressuredata = a.retrieve_pressure_data(1618365600, 2, sensor_ids)
    print("Pressure data in the timeslot:", pressuredata)
    all_pressuredata=a.retrieve_all_pressure_data("sensor_6")
    print("All pressure data per sensor_6:", all_pressuredata)

    #---------------------FLOW DATA --------------------------------------------
    sensor_ids= ['sensor1', 'sensor2']
    flowdata = a.retrieve_flow_data(1618365600, 2, sensor_ids)
    print("Flow data in the timeslot:", flowdata)
    all_flowdata=a.retrieve_all_flow_data("sensor1")
    print("All flow data per sensor1:", all_flowdata)
