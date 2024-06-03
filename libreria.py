from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
from collections import defaultdict
class utility():

    def __init__(self):
        try:
            self.client = MongoClient('mongodb+srv://admin_teamC:yGkHcCGezKMa1VET@waterleakage.zkwupj6.mongodb.net/?retryWrites=true&w=majority')
            self.db = self.client["Data"]
            self.user_collection = self.db["user_data"]
            self.pressure_collection = self.db["pressure_data"]
            self.flow_collection = self.db["flow_data"]
            self.leak_collection=self.db["leakages_data"]

        except Exception as e:
            message = "An error occurred while running the service:PROVA_MSG" + str(e)
            raise Exception(message)

    def get_users_list(self):
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
            users_list.append(user["full_name"])
        return users_list

    def get_users_info(self,user_ids):
        users_info=[]
        for id in user_ids:
            filter = {"user_id": id}
            info = self.user_collection.find_one(filter, {"_id": 0, "user_id":1, "password": 1, "address": 1, "sensors.sensor_id": 1})
            users_info.append(info)
        return users_info

    
    def get_user_sensors(self, full_name):
        name, surname = full_name.split(' ', 1)        
        filter = {"name": name, "surname": surname}
        user_info = self.user_collection.find_one(filter, {"sensors.sensor_id": 1})
        
        if user_info:
            sensor_ids = [sensor["sensor_id"] for sensor in user_info.get("sensors", [])]
            return sensor_ids
        else:
            return None


    def old_get_user_sensors(self,user_ids):
        user_sensors=[]
        for id in user_ids:
            filter = {"user_id": id}
            info = self.user_collection.find_one(filter, {"_id": 0, "sensors.sensor_id": 1})
            sensor_ids = [sensor["sensor_id"] for sensor in info.get("sensors", [])]
            user_dict = {"user_id": id, "sensor_ids": sensor_ids}
            user_sensors.append(user_dict)
        return user_sensors

    def OLD_get_sensors_info(self,sensor_ids):
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
    
    def get_sensors_info(self, full_name):
        # Dividi il nome completo in nome e cognome
        name, surname = full_name.split(' ', 1)
        
        # Trova l'utente usando il nome e cognome
        user_filter = {"name": name, "surname": surname}
        user_info = self.user_collection.find_one(user_filter, {"_id": 0, "sensors": 1})
        
        if not user_info:
            return None  # Se l'utente non è trovato, restituisci None o un'altra indicazione appropriata
        
        sensors_by_dma = {}
        sensors = user_info.get("sensors", [])
        
        for sensor in sensors:
            sensor_id = sensor.get("sensor_id")
            dma = sensor.get("dma")
            
            if dma not in sensors_by_dma:
                sensors_by_dma[dma] = []
            
            if sensor_id not in sensors_by_dma[dma]:
                sensors_by_dma[dma].append(sensor_id)
        
        # Convert the dictionary to the desired format
        zones_list = [{'zone': dma, 'sensors': sensors} for dma, sensors in sensors_by_dma.items()]
        
        return zones_list





    
    def get_flow_per_sensors(self, sensor_ids):
        combined_data = defaultdict(dict)
        # Iterate over each sensor_id
        for sensor_id in sensor_ids:
            query = {"bn": sensor_id}
            result = self.flow_collection.find(query, {"bt": 1, "v": 1, "bn": 1})
            
            for document in result:
                timestamp = document.get("bt")
                value = document.get("v")
                combined_data[timestamp]["timestamp"] = timestamp
                combined_data[timestamp][f"{sensor_id}_flow"] = value
        # Convert the combined data to a list of dictionaries
        combined_data_list = list(combined_data.values())
        df = pd.DataFrame(combined_data_list)
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        return df
    '''
    def get_flow_range(self,initial_date, duration_days, sensor_ids):
        starting_date = datetime.strptime(initial_date,"%d/%m/%Y")
        end_date= starting_date+ timedelta(days=duration_days)
        starting_time_unix =datetime.timestamp(starting_date)
        end_time_unix= datetime.timestamp(end_date)

        query = {
            "$and": [
                {"bt": {"$gte": starting_time_unix}},
                {"bt": {"$lt": end_time_unix}},
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
    '''

    def get_flow_range(self, initial_date, duration_days):
        starting_date = initial_date.to_pydatetime()
        end_date = starting_date - timedelta(days=duration_days)
        starting_time_unix = datetime.timestamp(starting_date)
        end_time_unix = datetime.timestamp(end_date)
        query = {
            "$and": [
                {"bt": {"$gte": end_time_unix}},
                {"bt": {"$lt": starting_time_unix}}
            ]
        }
        
        # Eseguire la query sul database
        result = self.flow_collection.find(query, {"bt": 1, "v": 1, "bn": 1})
        data = []
        for document in result:
            sensor_id = document.get("bn")
            timestamp = document.get("bt")
            value = document.get("v")
            data.append([timestamp, sensor_id, value])
        
        # Creare un DataFrame
        df = pd.DataFrame(data, columns=["timestamp", "sensor_id", "value"])
        
        # Pivotare il DataFrame per avere una colonna per ogni sensore
        df_pivot = df.pivot_table(index="timestamp", columns="sensor_id", values="value").reset_index()
        
        # Restituire il DataFrame pivotato
        return df_pivot
    
    def get_flow_test(self, sensor_ids):
        combined_data = defaultdict(dict)
        
        for sensor_id in sensor_ids:
            query = {"bn": sensor_id}
            result = self.flow_collection.find(query, {"bt": 1, "v": 1, "bn": 1}).sort("bt", -1).limit(288)
            
            for document in result:
                timestamp = document.get("bt")
                value = document.get("v")
                combined_data[timestamp]["timestamp"] = timestamp
                combined_data[timestamp][f"{sensor_id}_pressure"] = value

        combined_data_list = list(combined_data.values())
        df = pd.DataFrame(combined_data_list)
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        return df


    def get_all_flow(self):
        combined_data = defaultdict(dict)
        result = self.flow_collection.find({}, {"bt": 1, "v": 1, "bn": 1})

        for document in result:
            sensor_id = document.get("bn")
            timestamp = document.get("bt")
            value = document.get("v")
            combined_data[timestamp]["timestamp"] = timestamp
            combined_data[timestamp][f"{sensor_id}_flow"] = value

        combined_data_list = list(combined_data.values())
        df = pd.DataFrame(combined_data_list)
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        return df

    def OLDget_pressure_per_sensors(self, sensor_ids):
        combined_data = defaultdict(dict)

        # Iterate over each sensor_id
        for sensor_id in sensor_ids:
            query = {"bn": sensor_id}
            result = self.pressure_collection.find(query, {"bt": 1, "v": 1, "bn": 1})
            
            for document in result:
                timestamp = document.get("bt")
                value = document.get("v")
                combined_data[timestamp]["timestamp"] = timestamp
                combined_data[timestamp][f"{sensor_id}_pressure"] = value

        # Convert the combined data to a list of dictionaries
        combined_data_list = list(combined_data.values())
        df = pd.DataFrame(combined_data_list)
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        return df
    

    def get_pressure_test(self, sensor_ids):
        combined_data = defaultdict(dict)
        
        for sensor_id in sensor_ids:
            query = {"bn": sensor_id}
            result = self.pressure_collection.find(query, {"bt": 1, "v": 1, "bn": 1}).sort("bt", -1).limit(288)
            
            for document in result:
                timestamp = document.get("bt")
                value = document.get("v")
                combined_data[timestamp]["timestamp"] = timestamp
                combined_data[timestamp][f"{sensor_id}_pressure"] = value

        combined_data_list = list(combined_data.values())
        df = pd.DataFrame(combined_data_list)
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        return df

    def get_pressure_per_sensors(self, sensor_ids):
        combined_data = defaultdict(dict)

        # Iterate over each sensor_id
        for sensor_id in sensor_ids:
            query = {"bn": sensor_id}
            result = self.pressure_collection.find(query, {"bt": 1, "v": 1, "bn": 1}).limit(4032)
            
            count = 0  # Counter to track number of values for each sensor
            for document in result:
                if count >= 4032:
                    break  # Stop if we have already retrieved 4032 values for this sensor
                timestamp = document.get("bt")
                value = document.get("v")
                combined_data[timestamp]["timestamp"] = timestamp
                combined_data[timestamp][f"{sensor_id}_pressure"] = value
                count += 1

        # Convert the combined data to a list of dictionaries
        combined_data_list = list(combined_data.values())
        df = pd.DataFrame(combined_data_list)
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        return df

    
    def get_pressure_range(self, initial_date, duration_days, sensor_ids):
        starting_date = datetime.strptime(initial_date,"%d/%m/%Y")
        end_date= starting_date+ timedelta(days=duration_days)
        starting_time_unix =datetime.timestamp(starting_date)
        end_time_unix= datetime.timestamp(end_date)

        query = {
            "$and": [
                {"bt": {"$gte": starting_time_unix}},
                {"bt": {"$lt": end_time_unix}},
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

    def get_pressure_range(self, initial_date, duration_days):
        starting_date = initial_date.to_pydatetime()
        end_date = starting_date - timedelta(days=duration_days)
        starting_time_unix = datetime.timestamp(starting_date)
        end_time_unix = datetime.timestamp(end_date)
        query = {
            "$and": [
                {"bt": {"$gte": end_time_unix}},
                {"bt": {"$lt": starting_time_unix}}
            ]
        }
        
        # Eseguire la query sul database
        result = self.pressure_collection.find(query, {"bt": 1, "v": 1, "bn": 1})
        data = []
        for document in result:
            sensor_id = document.get("bn")
            timestamp = document.get("bt")
            value = document.get("v")
            data.append([timestamp, sensor_id, value])
        
        # Creare un DataFrame
        df = pd.DataFrame(data, columns=["timestamp", "sensor_id", "value"])
        
        # Pivotare il DataFrame per avere una colonna per ogni sensore
        df_pivot = df.pivot_table(index="timestamp", columns="sensor_id", values="value").reset_index()
        
        # Restituire il DataFrame pivotato
        return df_pivot

        

    
    
    def get_all_pressure(self):
        combined_data = defaultdict(dict)
        result = self.pressure_collection.find({}, {"bt": 1, "v": 1, "bn": 1})

        for document in result:
            sensor_id = document.get("bn")
            timestamp = document.get("bt")
            value = document.get("v")
            combined_data[timestamp]["timestamp"] = timestamp
            combined_data[timestamp][f"{sensor_id}_pressure"] = value

        combined_data_list = list(combined_data.values())
        df = pd.DataFrame(combined_data_list)
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        return df
    

    def get_state_values(self, dma):
        pipeline = [
            {
                "$match": {
                    "dma": dma
                }
            },
            {
                "$group": {
                    "_id": "$dma",
                    "max_leak_id": {"$max": "$leak_id"}
                }
            },
            {
                "$lookup": {
                    "from": self.leak_collection.name,
                    "let": {"dma": "$_id", "max_leak_id": "$max_leak_id"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {"$eq": ["$dma", "$$dma"]},
                                        {"$eq": ["$leak_id", "$$max_leak_id"]}
                                    ]
                                }
                            }
                        },
                        {
                            "$project": {
                                "_id": 0,
                                "dma": 1,
                                "cnt_normal": 1,
                                "cnt_leak": 1,
                                "current_situation": 1
                            }
                        }
                    ],
                    "as": "leak_info"
                }
            },
            {
                "$unwind": "$leak_info"
            },
            {
                "$replaceRoot": {"newRoot": {"$mergeObjects": ["$leak_info", {"dma": "$_id"}]}}
            }
        ]

        result = list(self.leak_collection.aggregate(pipeline))
        return result[0] if result else None
        

#main di test
if __name__ == "__main__":
    a=utility()
    users=a.get_users_list()
    print(users)
    sensors=a.get_user_sensors(users[0])
    # print(sensors)
    # # info=a.get_sensors_info(users[0])
    # # b=a.get_pressure_per_sensors(sensors)
    # # print(b)
    # c=a.get_pressure_test(sensors)
    # print(c)
    d=a.get_state_values("DMA1")
    print(d)




    