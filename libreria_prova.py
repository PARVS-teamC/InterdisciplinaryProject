from pymongo import MongoClient
class utility():

    def __init__(self):
      try:
          self.client = MongoClient('mongodb+srv://admin_teamC:yGkHcCGezKMa1VET@waterleakage.zkwupj6.mongodb.net/?retryWrites=true&w=majority')
          self.db = self.client["User_data"]
          self.collection = db["User_data"]

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

        result = self.collection.find(filter={},projection=project)
        for user in result:
            users_list.append(user)
        return users_list

    def retrieve_usersinfo(self):
        users_info=[]
        user_ids = [user['user_id'] for user in self.retrieve_userslist()]
        for id in user_ids:
            filter = {"user_id": id}
            info = collection.find_one(filter, {"_id": 0, "user_id":1, "password": 1, "address": 1, "sensors.sensor_id": 1})
            users_info.append(info)
        return users_info

    def retrieve_password(self):
        user_pwd=[]
        user_ids = [user['user_id'] for user in self.retrieve_userslist()]
        for id in user_ids:
            filter = {"user_id": id}
            pwd = collection.find_one(filter, {"_id": 0, "user_id":1, "password": 1})
            user_pwd.append(pwd)
        return user_pwd

    def retrieve_usersensors(self):
        user_sensors=[]
        user_ids = [user['user_id'] for user in self.retrieve_userslist()]
        for id in user_ids:
            filter = {"user_id": id}
            info = collection.find_one(filter, {"_id": 0, "sensors.sensor_id": 1})
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
            info = collection.find_one(filter, projection)

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



#main di test
if __name__ == "__main__":
    a=utility()
    sensors=a.retrieve_usersensors()
    print(sensors)
    sensor_ids = [user['sensor_ids'] for user in sensors] 
    print(a.retrieve_sensorsinfo(sensor_ids[0]))# sensor_ids = ["sensor_11", "sensor_12", "sensor_13"]