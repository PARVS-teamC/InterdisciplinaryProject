import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras import layers
from matplotlib import pyplot as plt

def moving_zscore(df, window_size):
    df_norm = df.copy()
    for col in df_norm.columns:
        df_norm[col] = (df_norm[col] - df_norm[col].rolling(window=window_size,min_periods=1).mean()) / (df_norm[col].rolling(window=window_size,min_periods=1).std())
    df_norm = df_norm.dropna() 
    return df_norm

def zscore(df, df_train):
    training_mean = df_train.mean()
    training_std = df_train.std()

    df_norm = (df - training_mean) / training_std
    df_norm = df_norm.dropna() 
    return df_norm 

def segmentation(df, window_size):
    segments = []
    num_segments = len(df) - window_size + 1
    for i in range(num_segments):
        segment = df.iloc[i:i+window_size, :]
        segments.append(segment)
    return segments

def get_index(df,col_name_list):
    index_list=[]
    for name in col_name_list:
        index = df.columns.get_loc(name)
        index_list.append(index)
    return index_list

window_size = 8640
window_seg = 288
#sensor_list=['n1','n4','n31']
#DMA7
sensor_list=['n188','n163','n613']
df = pd.read_csv('InterdisciplinaryProject/Data/2019_SCADA_Pressures.csv', sep=';')
indexes= get_index(df, sensor_list)
for column in df.columns[1:]:
    df[column] = pd.to_numeric(df[column].str.replace(',', '.'))
#DMA1
'''df_train = df.iloc[:4033, indexes]
#df_test = df.iloc[4034:52000, indexes] 
df_test = df.iloc[4034:, indexes]'''
#DMA4
'''df_train = df.iloc[:4033, :]
df_test = df.iloc[61058:104834, :]'''

df_train = df.iloc[:4033, indexes]
df_test = df.iloc[4034:, indexes]
#Timeseries data (train) without anomalies
fig, ax = plt.subplots()
df_train[sensor_list].plot(legend=False, ax=ax)
plt.show()
#Timeseries data (test) with anomalies
fig, ax = plt.subplots()
df_test[sensor_list].plot(legend=False, ax=ax)
plt.show()

#df_train=df_train.drop('Timestamp', axis=1)

#df_train_value=moving_zscore(df_train,window_size)
df_train_value=zscore(df_train,df_train)
print("Number of training samples:", len(df_train_value))
fig, ax = plt.subplots()
df_train_value[sensor_list].plot(legend=False, ax=ax)
plt.show()

x_train=segmentation(df_train_value, window_seg)
x_train=np.array(x_train)


model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
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
        layers.Dense(288 * len(sensor_list), activation='linear'),
        layers.Reshape((288, len(sensor_list)))
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()
history = model.fit(
    x_train,
    x_train,
    epochs=20,
    batch_size=144,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss")
    ],
)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

x_train_pred = model.predict(x_train)
'''
indexes=get_index(df_train,sensor_list)'''
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()

threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)


plt.plot(x_train[0])
plt.plot(x_train_pred[0])
plt.show()

# plt.plot(x_train[0,:,0])
# plt.plot(x_train_pred[0,:,0])
# plt.show()
# plt.plot(x_train[0,:,1])
# plt.plot(x_train_pred[0,:,1])
# plt.show()
# plt.plot(x_train[0,:,2])
# plt.plot(x_train_pred[0,:,2])
# plt.show()

#df_test=df_test.drop('Timestamp', axis=1)
#df_test_value = moving_zscore(df_test,window_size)
df_test_value= zscore(df_test,df_train)
fig, ax = plt.subplots()
df_test_value.plot(legend=False, ax=ax)
plt.show()

x_test = segmentation(df_test_value, window_seg)
x_test = np.array(x_test)
# Get test MAE loss.
x_test_pred = model.predict(x_test)

test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
print(type(test_mae_loss))
print(test_mae_loss.shape)

#test_mae_loss = test_mae_loss.reshape((-1))
print(test_mae_loss.shape )
plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()


anomalies = test_mae_loss > threshold

print(anomalies)
#print("Number of anomaly samples: ", np.sum(anomalies))
#print("Indices of anomaly samples: ", np.where(anomalies))
import numpy as np

# Assuming anomalies is a 2D NumPy array
for i in range(anomalies.shape[1]):
    column_anomalies = anomalies[:, i]
    num_anomalies = np.sum(column_anomalies)
    anomaly_indices = np.where(column_anomalies)[0]
    print(f"Number of anomaly samples in column {i}: {num_anomalies}")
    print(f"Indices of anomaly samples in column {i}: {anomaly_indices}")


anomaly_indices_per_column = {}

# Iterate over the columns
for column in range(len(anomalies[0])):
    # Initialize an empty list to store anomaly indices for this column
    anomaly_indices_per_column[column] = []
    
    # Iterate over the data indices
    for data_idx in range(window_seg - 1, len(df_test_value) - window_seg + 1):
        # Check if the value in this column is an anomaly
        if anomalies[data_idx, column]:
            # Append the index to the list of anomaly indices for this column
            anomaly_indices_per_column[column].append(data_idx)

# Select subset of data containing only anomalous samples for each column
df_subset_per_column = {}
for column, indices in anomaly_indices_per_column.items():
    df_subset_per_column[column] = df_test_value.iloc[indices]

# Plot the original data and the subset of anomalous data for each column
fig, axes = plt.subplots(len(df_test_value.columns), 1, figsize=(10, 5 * len(df_test_value.columns)), sharex=True)
for i, column_name in enumerate(df_test_value.columns):
    ax = axes[i]
    df_test_value[column_name].plot(legend=False, ax=ax)
    df_subset_per_column[i][column_name].plot(legend=False, ax=ax, color="r")
    ax.set_title(f"Column: {column_name}")
plt.show()
'''anomalous_data_indices = []
for data_idx in range(window_seg - 1, len(df_test_value) - window_seg + 1):
    if np.all(anomalies[data_idx - window_seg + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)


df_subset = df_test[sensor_list].iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df_test[sensor_list].plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r")
plt.show()
'''