import numpy as np
import pandas as pd
import keras
from keras import layers
from matplotlib import pyplot as plt

def moving_zscore(df, window_size):
    df_norm = df.copy()
    for col in df_norm.columns:
        df_norm[col] = (df_norm[col] - df_norm[col].rolling(window=window_size,min_periods=1).mean()) / (df_norm[col].rolling(window=window_size,min_periods=1).std())
    df_norm = df_norm.dropna() 
    return df_norm

def segmentation(df, window_size):
    segments = []
    num_segments = len(df) - window_size + 1
    for i in range(num_segments):
        segment = df.iloc[i:i+window_size, :]
        segments.append(segment)
    return segments

window_size = 8640
window_seg = 288
df_dataset = pd.read_csv('Data/2019_SCADA_Pressures.csv', sep=';')
for column in df_dataset.columns[1:]:
    df_dataset[column] = pd.to_numeric(df_dataset[column].str.replace(',', '.'))
df_train = df_dataset.iloc[:4033, :4]
df_test = df_dataset.iloc[4034:52000, :4]
#Timeseries data (train) without anomalies
fig, ax = plt.subplots()
df_train.plot(legend=False, ax=ax)
plt.show()
#Timeseries data (test) with anomalies
fig, ax = plt.subplots()
df_test.plot(legend=False, ax=ax)
plt.show()

df_train=df_train.drop('Timestamp', axis=1)

df_train_value=moving_zscore(df_train,window_size)
print("Number of training samples:", len(df_train_value))
fig, ax = plt.subplots()
df_train_value.plot(legend=False, ax=ax)
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
        layers.Dense(288 * 3, activation='linear'),
        layers.Reshape((288, 3))
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()
history = model.fit(
    x_train,
    x_train,
    epochs=25,
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

df_test=df_test.drop('Timestamp', axis=1)
df_test_value = moving_zscore(df_test,window_size)
fig, ax = plt.subplots()
df_test_value.plot(legend=False, ax=ax)
plt.show()

x_test = segmentation(df_test_value, window_seg)
x_test = np.array(x_test)
# Get test MAE loss.
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
#test_mae_loss=test_mae_loss[:, :3]
test_mae_loss = test_mae_loss.reshape((-1))

plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

anomalous_data_indices = []
for data_idx in range(window_seg - 1, len(df_test_value) - window_seg + 1):
    if np.all(anomalies[data_idx - window_seg + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)

df_subset = df_test.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df_test.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r")
plt.show()