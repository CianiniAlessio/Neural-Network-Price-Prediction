import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from sklearn.linear_model import LogisticRegression
from keras.layers import LSTM, Dense, Dropout
import keras
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
import config




# MANAGE DATA FRAME TO ADD RATIO BETWEEN VOLUME OF T-1 AND T-2

client = Client(config.api_key,config.api_secret)
print("Downloading data...")
trades = client.get_historical_klines("BTCUSDT", client.KLINE_INTERVAL_15MINUTE, "3 Sep, 2022", "4 Oct, 2022")
print("Downloaded data..")
dataframe = pd.DataFrame()
dataframe['Open'] = [(float)(i[1]) for i in trades]
dataframe['Close'] = [(float)(i[4]) for i in trades]
dataframe['Volume'] = [(float)(i[5]) for i in trades]

def Exp_m_a(price, last_ema,days):

    result = last_ema+2*(price-last_ema)
    return result

def manage_data_frame(df):
    zeros = np.zeros(len(df['Volume']))
    df['Volume Ratio'] = zeros
    for i in range(2,len(df['Volume'])):
        df.at[i,'Volume Ratio'] = (float)((float)(df.at[i-1,'Volume'])/(float)(df.at[i-2,'Volume'])-1)   

    return df

# READ THE CSV FOR BTC FROM YAHOO FINANCE

df = pd.read_csv('BTC-USD.csv')
df = dataframe
df = manage_data_frame(df)

# ADDING THE COLUMN FOR THE EMA 

length = 20
df["EMA_"+str(length)] = ta.ema(df['Close'], length=length)

# DECIDING WHICH WILL BE THE INPUT OF MY NEURAL NETWORK

X_1 = df['Open'].loc[length+2:]
X_2 = df['Volume Ratio'].loc[length+2:]
X_3=  df["EMA_"+str(length)].loc[length+2:]

# SELECT THE OUTPUT

Y = df['Close'].loc[length+2:]


#====== CREATING THE MODEL ======

x1 = Input(shape =(1,))
x2 = Input(shape =(1,))
x3 = Input(shape =(1,))


input_layer = concatenate([x1,x2,x3])
hidden_layer = Dense(units=6, activation='relu')(input_layer)
hidden_layer2 = Dense(units=12, activation='relu')(hidden_layer)
hidden_layer3 = Dense(units=6, activation='relu')(hidden_layer2)
prediction = Dense(1, activation='linear')(hidden_layer3)
model = Model(inputs=[x1, x2, x3], outputs=prediction)

model.compile(loss="mean_squared_error", 
              optimizer='adam', 
              metrics=['mae'])

model.fit([X_1, X_2, X_3], Y, epochs = 100, batch_size=32, verbose=2)


model.save("C:\\Users\\Administrator\\OneDrive\\Desktop\\python\\neural network\\saved_model\\myModel")
 

model = tf.keras.models.load_model("saved_model\\myModel")
# STATIC VARIABLE EXCPET FOR PRICE ( FOR NOW WILL CHANGE )
volume_ratio = df.loc[len(df['Volume Ratio'])-1].at['Volume Ratio']
ema = df.loc[len(df['EMA_'+str(length)])-1].at['EMA_'+str(length)]
print("FIRST_VOLUME : {}".format(volume_ratio))
print("FIRST_EMA : {}".format(ema))

# RETRIEVING DATA FROM THE WEB OF BTC, IF THE MODEL PREDICT A LOWER PRICE THAT MEANS WE ARE GOING TO SELL
# THE OPPOSITE WE ARE GOING TO BUY
# WE'RE GOING TO CHECK THE PROFIT EVERY 'X' SECONDS, JUST CHANGE IT

percentage = 0
commission = 0.0002
volume = 0
while True:
    
    # GETTING THE BID
    #bid = requests.get(url = "https://api.cryptowat.ch/markets/binance/btcusdt/orderbook?limit=1").json()['result']['bids'][0][0]
    
    last_starting = (float)(client.get_symbol_ticker(symbol = "BTCUSDT")['price'])
    
    
    #GETTIN THE VOLUME TO CALCULATE THE RATIO AS THIRD INPUT
    
    response_volume = client.get_historical_klines("BTCUSDT", client.KLINE_INTERVAL_15MINUTE)
    
    # LEN()-2 MEANS TWO END OF CANDLES BEFORE
    
    volume = (float)((response_volume)[len(response_volume)-2][5])
    new_volume = (float)((response_volume)[len(response_volume)-1][5])

    volume_ratio = (float)((new_volume/volume) - 1)
    
    print("Volume Ratio: {}".format(volume_ratio))    
    
    #CALCULATE SECOND INPUT (EXPONENTIAL MOVING AVERAGE)
    
    ema = Exp_m_a(last_starting,ema,length)
    
    print("Ema : {}".format(ema))
    
    X_1 = np.array([(float)(last_starting)])
    X_2 = np.array([(float)(volume_ratio)])
    X_3 = np.array([(float)(ema)])

    # PREPARING THE INPUT 

    k = [X_1,X_2,X_3]
    
    #PREDICT THE PRICE
    
    pred = model.predict(k)[0][0]
    print("last : {} - prediction : {}".format(last_starting,pred))

    # WAITING 
    
    time.sleep(2)

    #GETTING THE NEW PRICE

    last_final =(float)(client.get_symbol_ticker(symbol = "BTCUSDT")['price'])
    diff = abs(pred-(float)(last_final))
    
    # CHECKING PROFIT

    if(last_starting>pred):
        #sell
        print("\n=== SELL ===\n\nInitial Price :{}\nPrediction: {}\nFinal Price: {}\nProfit:\t\t {}\nDifference: {}".format(last_starting,pred,last_final,last_starting-last_final,diff))
        percentage += (last_starting/last_final-1) - commission
        print("PROFIT PERCENTAGE :\t\t\t\t {}%\n".format(round(percentage*100,5)))
    else:
        print("\n=== BUY ===\n\nInitial Price :{}\nPrediction: {}\nFinal Price: {}\nProfit:\t\t {}\nDifference: {}".format(last_starting,pred,last_final,last_final-last_starting,diff))
        percentage += (last_starting/last_final-1) - commission
        print("PROFIT PERCENTAGE :\t\t\t\t {}%\n".format(round(percentage*100,5)))


