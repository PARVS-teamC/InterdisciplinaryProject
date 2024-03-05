import pandas as pd
from scipy.stats import zscore
from keras.layers import Input, Conv1D, Conv1DTranspose, AveragePooling1D, Activation,Dense,Reshape,Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.activations import tanh, linear
import numpy as np
import matplotlib.pyplot as plt

def moving_zscore(df, window_size):
    df_norm = df.copy()
    for col in df_norm.columns[1:]:
        df_norm[col] = (df_norm[col] - df_norm[col].rolling(window=window_size,min_periods=1).mean()) / (df_norm[col].rolling(window=window_size,min_periods=1).std()+0.0001)
    df_norm = df_norm.dropna() 
    return df_norm

def segmentation(df, window_size):
    segments = []
    num_segments = len(df) - window_size + 1
    for i in range(num_segments):
        segment = df.iloc[i:i+window_size, :]
        segments.append(segment)
    return segments

def compute_cumulative_dma_error(reconstructed_data, sensor_indices,test_data):
    ns = len(sensor_indices)
    num_segments = reconstructed_data.shape[0]
    num_timestamps = reconstructed_data.shape[1]
    
    dma_errors = np.zeros(num_timestamps)

    for j in range(num_timestamps):
        sre_per_timestamp = []
        for seg_idx in range(num_segments):
            reconstructed_data_j = reconstructed_data[seg_idx, j, :]
            sre = np.zeros(ns)
            for idx, sensor_idx in enumerate(sensor_indices):
                sensor_data = reconstructed_data_j[sensor_idx]
                sre[idx] = np.mean(np.abs(sensor_data - test_data[seg_idx, j, sensor_idx]))
            sre_per_timestamp.append(np.sum(sre))
        
        # Compute median reconstruction error for current timestamp
        median_sre = np.median(sre_per_timestamp)
        dma_errors[j] = median_sre
    
    return dma_errors


def oneD_CNN(data,test_data):

    encoder = tf.keras.models.Sequential([
        Conv1D(filters=32, kernel_size=7, strides=1, activation='tanh'),
        AveragePooling1D(pool_size=2),
        Conv1D(filters=16, kernel_size=7, strides=1, activation='tanh'),
        AveragePooling1D(pool_size=2)
    ])

    # Decoder
    decoder = tf.keras.models.Sequential([
        Conv1DTranspose(filters=16, kernel_size=7, strides=4, activation='tanh'),
        AveragePooling1D(pool_size=2),
        Conv1DTranspose(filters=32, kernel_size=7, strides=4, activation='tanh'),
        AveragePooling1D(pool_size=2),
        Flatten(),
        Dense(288 * 33, activation='linear'),
        Reshape((288, 33))
    ])

    autoencoder = tf.keras.models.Sequential([
        encoder,
        decoder
    ])

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mse', 'mae'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True)

    autoencoder.fit(data, data, batch_size=144, epochs=14, validation_split=0.2, callbacks=[early_stopping])

    autoencoder.summary()
    #evaluation=autoencoder.evaluate(test_data, test_data)
    #print("Evaluation Loss:", evaluation)
    reconstructed_data = autoencoder.predict(test_data)
    
    return reconstructed_data

if __name__ == '__main__':
    window_size = 8640
    window_seg = 288
    df = pd.read_csv('Data/2019_SCADA_Pressures.csv', sep=';')#('/content/2019_SCADA_Pressures.csv', sep=';')
    df_train = df[:4033]
    df_test= df[4034:50000]
    for col in df_train.columns[1:]:
        df_train.loc[:, col] = pd.to_numeric(df_train[col].str.replace(',', '.')).astype('float32')
        df_test.loc[:, col] = pd.to_numeric(df_test[col].str.replace(',', '.')).astype('float32')
    #train moving score    
    df_norm_train = moving_zscore(df_train, window_size)
    df_norm_train.drop(columns=df.columns[0], axis=1, inplace=True) #remove timestamp
    #same on test
    df_norm_test = moving_zscore(df_test, window_size)
    df_norm_test.drop(columns=df.columns[0], axis=1, inplace=True)
    #train segmentatiom
    segments_train = segmentation(df_norm_train, window_seg)
    array_list = [df.values for df in segments_train]
    input_data = np.stack(array_list, axis=2)
    input_data = np.transpose(input_data, (2, 0, 1))
    print("Shape of input data array:", input_data.shape)
    #test segmentation
    segments_test = segmentation(df_norm_test, window_seg)
    array_list_te = [df.values for df in segments_test]
    test_data = np.stack(array_list_te, axis=2)
    test_data = np.transpose(test_data, (2, 0, 1))
    print("Shape of input data array:", test_data.shape)
    #nn
    input_data = np.asarray(input_data).astype(np.float32)
    test_data = np.asarray(test_data).astype(np.float32)
    reconstructed_data = oneD_CNN(input_data,test_data)
    # Compute cumulative DMA error
    sensor_indices_one = [0, 1, 2]  # n1, n4, n31 as sensor indices for zone 1
    cdma_error = compute_cumulative_dma_error(reconstructed_data, sensor_indices_one,test_data)
    print("Cumulative DMA Error shape:", cdma_error.shape)
