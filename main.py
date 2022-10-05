import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from sklearn.linear_model import LogisticRegression
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras import Input  
from tensorflow.keras import Model 
from tensorflow.keras.layers import *
from sklearn import preprocessing
import pandas_ta as ta
from binance import Client
import requests
import time


def manage_data_frame(df):
    zeros = np.zeros(len(df['Volume']))
    df['Volume Ratio'] = zeros
    for i in range(2,len(df['Volume'])):
        df.at[i,'Volume Ratio'] = df.at[i-1,'Volume']/df.at[i-2,'Volume'] - 1
        

    return df


df = pd.read_csv('BTC-USD.csv')
df = manage_data_frame(df)
length = 20
df["EMA_"+str(length)] = ta.ema(df['Close'], length=length)





X_1 = df['Open'].loc[length+2:]
X_2 = df['Volume Ratio'].loc[length+2:]
X_3=  df["EMA_"+str(length)].loc[length+2:]
# column = 'Volume'
# X_2 = (df[column] - df[column].min()) / (df[column].max() - df[column].min())    


Y = df['Adj Close'].loc[length+2:]

x1 = Input(shape =(1,))
x2 = Input(shape =(1,))
x3 = Input(shape =(1,))


input_layer = concatenate([x1,x2,x3])
hidden_layer = Dense(units=12, activation='relu')(input_layer)
hidden_layer2 = Dense(units=18, activation='relu')(hidden_layer)
hidden_layer3 = Dense(units=14, activation='relu')(hidden_layer2)
prediction = Dense(1, activation='linear')(hidden_layer3)
model = Model(inputs=[x1, x2, x3], outputs=prediction)

model.compile(loss="mean_squared_error", 
              optimizer='adam', 
              metrics=['mae'])

model.fit([X_1, X_2, X_3], Y, epochs = 200, 
          batch_size=32, verbose=2)

# X_1 = np.array([225])
# X_2 = np.array([0.5])
# X_3 = np.array([217])
# k = [X_1,X_2,X_3]

volume_ratio = df.loc[len(df['Volume Ratio'])-1].at['Volume Ratio']
ema = df.loc[len(df['EMA_'+str(length)])-1].at['EMA_'+str(length)]
while True:
    bid = requests.get(url = "https://api.cryptowat.ch/markets/binance/btcusdt/orderbook?limit=1").json()['result']['bids'][0][0]
    X_1 = np.array([bid])
    X_2 = np.array([volume_ratio])
    X_3 = np.array([ema])
    k = [X_1,X_2,X_3]
    pred = model.predict(k)
    time.sleep(300)
    ask = requests.get(url = "https://api.cryptowat.ch/markets/binance/btcusdt/orderbook?limit=1").json()['result']['asks'][0][0]
    diff = abs(pred-ask)
    if(bid>pred):
        #sell
        print("=== SELL ===\nInitial Price :{}\nPrediction: {}\nFinal Price: {}\nProfit: {}\nDifference: {}".format(bid,pred,ask,bid-ask,diff))
    else:
        print("=== BUY ===\nInitial Price :{}\nPrediction: {}\nFinal Price: {}\nProfit: {}\nDifference: {}".format(bid,pred,ask,ask-bid,diff))
# for layer in model.layers:
#     print(layer.weights)



