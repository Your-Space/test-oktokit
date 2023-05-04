import time
import numpy as np
import pandas as pd
import tensorflow
import os 
import requests
import json

os.environ["CUDA_VISIBLE_DEVICES"]="0"
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import load_model

from keras.metrics import RootMeanSquaredError, MeanAbsoluteError

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tensorflow.config.experimental.set_memory_growth(gpu, True)

import tensorflow as tf
''
horizon = 150  # This varies based on the model and is the input window.

def get_1mdata_by_identifier(identifier):
    a = requests.get('https://dlrs31datasync.eaift.com/Training/GetAggregates?Identifier='+ identifier +'&WithHeaders=true&Fields=Close&Fields=Volume&Fields=Transactions&Multiplier=1&Timespan=minute')
    return json.loads(a.content)

def process_api_response(json_data):
    result_dict = {result['fieldName']: result['values'] for result in json_data['result']['single']['results']}
    response = {key: result_dict[key] for key in json_data['result']['single']['fields']}
    return pd.DataFrame(data=response)

print('hello')
xbtcusd = get_1mdata_by_identifier('XBTCUSD')
print('hello 2')
data = process_api_response(xbtcusd)

sc = MinMaxScaler(feature_range=(0, 1))

# Add more market data here

data_scaled = sc.fit_transform(data)

x_train_scaled = []
for i in range(horizon, len(data_scaled)):
    x_train_scaled.append(data_scaled[i-horizon:i])

# Convert data into proper format for training
x_train = np.array(x_train_scaled)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], -1))

x_train = np.float32(x_train)
print(x_train[0:1])
print('lskdjfalkdsjf')


# model = tf.keras.models.load_model('./btcusdt-d_3-h_150-l_180-epoch_13-loss_0.000062247-mse_0.007889677_checkpoint/')
model = tf.keras.models.load_model('D:/eudaimonia ml/prediction_example/btcusdt-d_3-h_150-l_180-epoch_42-loss_0'
                                   '.000062199-mse_0.007886650_checkpoint')

result = model.predict(x_train[0:1])

# Another example which processes multiple timesteps:

results = model.predict(x_train[0:15])
print(results)