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
scalers={}


def transform_data(data):

    '''
    Preprocessed data is transformed into the form wuch that it is used by model.
    Minmax scaling, training test split.

    Input
    -----
    data: preprocessed data

    Output
    -----
    transformed data, xtrain, ytrain, xtest, ytest

    '''

    df2 = data[["date","hour","CLOUD_COVERAGE","TEMPERATURE","WIND_SPEED","PRECIPITATION"]]
    df2["no_of_orders"] = 1

    list_dates = df2.date.unique()

    df2 = utility.fill_all_hour_data(df2, list_dates, ismultivariate=True)

    multivariate_data = df2.groupby(["date","hour"]).aggregate({"no_of_orders":np.sum,"CLOUD_COVERAGE":np.mean,"TEMPERATURE":np.mean,"WIND_SPEED":np.mean,"PRECIPITATION":np.mean})

    multivariate_data = multivariate_data.reset_index().drop(columns=["date","hour"])

    multivariate_data["CLOUD_COVERAGE"].fillna(np.mean(multivariate_data["CLOUD_COVERAGE"]), inplace=True)
    multivariate_data["TEMPERATURE"].fillna(np.mean(multivariate_data["TEMPERATURE"]), inplace=True)
    multivariate_data["WIND_SPEED"].fillna(np.mean(multivariate_data["WIND_SPEED"]), inplace=True)


    
    for i in multivariate_data.columns:
        scaler = MinMaxScaler(feature_range=(-1,1))
        s_s = scaler.fit_transform(multivariate_data[i].values.reshape(-1,1))
        s_s=np.reshape(s_s,len(s_s))
        scalers['scaler_'+ i] = scaler
        multivariate_data[i]=s_s

    multivariate_data = np.array(multivariate_data)

    train_data, training_size, test_data, test_size = utility.train_test_split_multivariate(multivariate_data,0.65)

    time_step = 10
    X_train, y_train = utility.dataset_creation(train_data, time_step, ismultivariate=True)
    X_test, ytest = utility.dataset_creation(test_data, time_step, ismultivariate=True)

    return multivariate_data, X_train, y_train, X_test, ytest


def model_structure(X_train, y_train, X_test, ytest):


    '''
    Defining the model architecture
    
    '''

    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='mean_squared_error',optimizer=opt)

    ## fitting the model
    hist =  model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)
    utility.save_json(os.path.join(json_dir,'history_multivariate.json'), hist.history)
    model.save(os.path.join(model_dir,"LSTMN_MULTIVARIATE.h5"))

    ## Prediction using model
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    ##Transformback to original form
    train_predict=scalers['scaler_no_of_orders'].inverse_transform(train_predict)
    test_predict=scalers['scaler_no_of_orders'].inverse_transform(test_predict)

    return train_predict, test_predict


def plot_output(multivariate_data, train_predict, test_predict):

    '''
    Plotting the evaluation and the results
    
    '''

    plt.figure(figsize=(30,10))
    look_back=10
    trainPredictPlot = np.empty_like(multivariate_data)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(multivariate_data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(multivariate_data)-1, :] = test_predict
    # plot baseline and predictions
    plt.plot(scalers['scaler_no_of_orders'].inverse_transform(multivariate_data[:,0].reshape(-1,1)))
    plt.plot(testPredictPlot)
    plt.legend(["Actual","Test"])
    plt.show()