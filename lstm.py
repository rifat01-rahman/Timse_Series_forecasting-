import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler,StandardScaler


df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv")


df['date'] = pd.to_datetime(df['date'])

df.set_index('date')[['Appliances', 'lights','T_out', 'RH_1', 'Visibility']].plot(subplots=True)


df_input=df[['Appliances','T_out', 'RH_1', 'Visibility']]

# Data Scaling

scaler = MinMaxScaler()
data_scaled=scaler.fit_transform(df_input)


features = data_scaled
target = data_scaled[:,0]


TimeseriesGenerator(features,target,length=2,sampling_rate=1,batch_size=1)[0]

# Data Splitting

xtrain,xtest,ytrain,ytest=train_test_split(features,target,test_size=0.20,random_state=123,shuffle=False)



win_length=720
batch_size=32
num_features=4
train_generator = TimeseriesGenerator(xtrain, ytrain, length=win_length, sampling_rate=1, batch_size=batch_size)
test_generator = TimeseriesGenerator(xtest, ytest, length=win_length, sampling_rate=1, batch_size=batch_size)

# Model Architecture 


model = tf.keras.Sequential([
    tf.keras.layers.LSTM(
        128,
        return_sequences=True,
        input_shape=(win_length, num_features)
    ),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(
        64,
        return_sequences=True
    ),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(
        32,
        return_sequences=False
    ),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])



early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    restore_best_weights=True
)

model.compile(
    loss=tf.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.metrics.MeanAbsoluteError()]
)

history = model.fit(
    train_generator,
    epochs=1,
    validation_data=test_generator,
    shuffle=False,
    callbacks=[early_stopping]
)

# Model Evaluation

model.evaluate(
    test_generator,
    verbose=0
)


xtest[:,1:][win_length:]

df_pred=pd.concat([pd.DataFrame(predictions), pd.DataFrame(xtest[:,1:][win_length:])],axis=1)

rev_trans=scaler.inverse_transform(df_pred)


df_final=df_input[predictions.shape[0]*-1:]

df_final['App_Pred']=rev_trans[:,0]

# Final Plot
df_final[['Appliances','App_Pred']].plot()