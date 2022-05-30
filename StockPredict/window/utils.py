import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from StockPredict.settings import BASE_DIR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os



def save_img():
    data = pd.read_csv(BASE_DIR/'adrd.us.csv').fillna(0)
    data = data.sort_index(ascending=False)
    data.index = range(len(data))
    data.head()




    dateparse = lambda dates:pd.datetime.strptime(dates, '%Y/%m/%d')
    data = pd.read_csv(BASE_DIR/'adrd.us.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
    data = data.sort_index(ascending=True)
    data.info()



    ma_list = [5,10,20,50,100,200]
    for ma in ma_list:
        data['Open_ma_' + str(ma)] = data['Open'].rolling(ma).mean()
        data['High_ma_' + str(ma)] = data['High'].rolling(ma).mean()
        data['Low_ma_' + str(ma)] = data['Low'].rolling(ma).mean()
        data['Close_ma_' + str(ma)] = data['Close'].rolling(ma).mean()
    data=data.dropna()
    # print(data)





    plt.figure(figsize=(16,8))
    plt.plot(data['Close'], color = 'green', label = 'Close Stock Price')
    plt.plot(data['Open_ma_5'], color = 'red', label = 'ma_5')
    plt.plot(data['Open_ma_10'], color = 'blue', label = 'ma_10')
    plt.plot(data['Open_ma_20'], color = 'green', label = 'ma_20')
    plt.plot(data['Open_ma_50'], color = 'yellow', label = 'ma_50')
    plt.plot(data['Open_ma_100'], color = 'black', label = 'ma_100')
    plt.plot(data['Open_ma_200'], color = 'magenta', label = 'ma_200')
    plt.title('ADRD.US')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    # plt.show()
    # fig = plt.plot.get_figure()
    plt.savefig(os.path.join("window/static/","output1.png"))



    train_data, test_data = data[0:int(len(data)*0.8)], data[int(len(data)*0.8):int(len(data))]



    plt.figure(figsize=(16, 8))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Close')
    plt.plot(data['Close'], 'green', label='Train data')
    plt.plot(test_data['Close'], 'blue', label='Test data')
    plt.legend()
    # fig2 = plt.get_figure()
    plt.savefig(os.path.join("window/static/","output2.png"))



    train = train_data.iloc[:, 0:1].values 
    test = test_data.iloc[:, 0:1].values


    scaler = MinMaxScaler()# normalize the dataset
    train_scaled = scaler.fit_transform(train)



    timesteps = 7
    X_train = []
    y_train = []
    for i in range(timesteps, train.shape[0]):
        X_train.append(train_scaled[i-timesteps:i, 0]) 
        y_train.append(train_scaled[i, 0]) 
    X_train, y_train = np.array(X_train), np.array(y_train)
    # print("X_train shape--",X_train.shape)
    # print("Y_train shape--",y_train.shape)





    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  
    # print("X_train samples--",X_train.shape[0])
    # print("X_train timesteps--",X_train.shape[1])



    from numpy.random import seed
    seed(2022)



    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.20))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.25))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.25))

    # Adding the output layer
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Training set
    history=model.fit(X_train, y_train, epochs = 40, batch_size = 32,validation_split=0.3)

    model.summary()


    plt.rcParams["figure.figsize"] = (20,5)
    plt.plot(history.history[ 'loss' ]) 
    plt.plot(history.history[ 'val_loss' ]) 
    plt.title( 'model train vs validation loss' ) 
    plt.ylabel( 'loss' ) 
    plt.xlabel( 'epoch' ) 
    plt.legend([ 'train' , 'validation' ], loc= 'upper right' ) 
    # plt.show()
    # fig3 = plt.get_figure()
    plt.savefig(os.path.join("window/static/","output3.png"))


    test_data

    real_stock_price = test_data.iloc[:, 3:4].values 


    # combine original train and test data vertically
    # as previous Open Prices are not present in the test dataset
    # e.g. for predicting Open price for first date in the test data, we will need stocl open prices on timesteps previous dates
    combine = pd.concat((train_data['Close'], test_data['Close']), axis = 0)


    # our test inputs also contains stock open Prices of last timesteps dates (as described above)
    test_inputs = combine[len(combine) - len(test_data) -timesteps:].values


    test_inputs  = test_inputs.reshape(-1, 1)
    test_inputs = scaler.transform(test_inputs)

    test_inputs.shape

    X_test = []
    for i in range(timesteps, test_data.shape[0]+timesteps):
        X_test.append(test_inputs[i-timesteps:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    # inverse_transform because prediction is done on scaled inputs
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)





    plt.figure(figsize=(16,8))
    plt.plot(data.index[-600:], data['Close'].tail(600), color = 'green', label = 'Test Stock Price')
    plt.plot(test_data.index, real_stock_price, color = 'red', label = 'Real Stock Price')
    plt.plot(test_data.index, predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("window/static/","output4.png"))

    mse = mean_squared_error(real_stock_price, predicted_stock_price)
    print('MSE: '+str(mse))
    mae = mean_absolute_error(real_stock_price, predicted_stock_price)
    print('MAE: '+str(mae))
    rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    print('RMSE: '+str(rmse))
