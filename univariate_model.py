import numpy as np
import pandas as pd
import os
# import osmnx as ox
import datetime
import sklearn
import matplotlib.pyplot as plt
import geopy
import json
import ast
import utility
import math
import geopy.distance as gd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow


model_dir = 'models'
json_dir = 'jsons'
scaler=MinMaxScaler(feature_range=(0,1))


def transform_data(data):

    df1 = data[["date","hour"]]
    df1["no_of_orders"] = 1
    list_dates = df1.date.unique()
    df1 = utility.fill_all_hour_data(df1, list_dates)

    df1.groupby(["date", "hour"]).sum()
    df_hourly_orders = df1.groupby(["date", "hour"]).sum().reset_index()["no_of_orders"]

    ## plotting figure
    plt.figure(figsize=(30,15))
    plt.plot(df_hourly_orders)
    plt.title("All orders plot grouped by date and hour")

    # scaler=MinMaxScaler(feature_range=(0,1))
    df_hourly_orders=scaler.fit_transform(np.array(df_hourly_orders).reshape(-1,1))

    train_data, training_size, test_data, test_size = utility.train_test_split_(df_hourly_orders,0.65)

    time_step = 10
    X_train, y_train = utility.dataset_creation(train_data, time_step)
    X_test, ytest = utility.dataset_creation(test_data, time_step)

    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    return df_hourly_orders, X_train, y_train, X_test, ytest

def model_structure(X_train, y_train, X_test, ytest):


    ## model building
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    ## fitting the model
    hist =  model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)
    utility.save_json(os.path.join(json_dir,'history.json'), hist.history)
    model.save(os.path.join(model_dir,"LSTMN_UNIVARIATE.h5"))

    ## Prediction using model
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

    return train_predict, test_predict


def plot_output(df_hourly_orders, train_predict, test_predict):



    plt.figure(figsize=(30,10))
    look_back=10
    trainPredictPlot = np.empty_like(df_hourly_orders)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df_hourly_orders)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df_hourly_orders)-1, :] = test_predict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(df_hourly_orders))
    plt.plot(testPredictPlot)
    plt.legend(["Actual","Test"])
    plt.show()
