from random import shuffle
import pandas as pd
import numpy as np
import json
import geopy.distance as gd


def calculate_dist_user_venue(coord1, coord2):

    '''
    Calculating the distance between user and venue location
    '''

    distance_geopy = gd.distance(coord1, coord2).km
    return distance_geopy


def fill_all_hour_data(data, list_dates, ismultivariate=False):

    '''
    Processing the data, filling the all hour data
    '''

    if ismultivariate:

        for dat_ in list_dates:
            for time in ["4","5", "6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22"]:
                if ( (data["date"]==dat_) & (data["hour"]==int(time)) ).any()==False:
                    # data = data.append({"date":dat_,"hour":int(time),"no_of_orders":0}, ignore_index=True)

                    data_new_row = pd.DataFrame({"date":[dat_],"hour":[int(time)],"no_of_orders":[0],"CLOUD_COVERAGE":[0],"TEMPERATURE":[0],"WIND_SPEED":[0],"PRECIPITATION":[0]})
                    data = pd.concat([data, data_new_row])

        return data



    for dat_ in list_dates:
        for time in ["4","5", "6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22"]:
            if ( (data["date"]==dat_) & (data["hour"]==int(time)) ).any()==False:
                # data = data.append({"date":dat_,"hour":int(time),"no_of_orders":0}, ignore_index=True)

                data_new_row = pd.DataFrame({"date":[dat_],"hour":[int(time)],"no_of_orders":[0]})
                data = pd.concat([data, data_new_row])

    return data


def train_test_split_(data, ratio, istest=False):

    if istest:

        # data = shuffle(data)
        training_size= int(len(data)*0.65)
        val_size = len(data)-training_size -10
        test_size = 10
        train_data,val_data,test_data = data[0:training_size,:],data[training_size:len(data)-10,:1], data[len(data)-10:,:1]

        return train_data, training_size, val_data, val_size, test_data, test_size


    # data = shuffle(data)
    training_size= int(len(data)*0.65)
    test_size= len(data)-training_size
    train_data,test_data= data[0:training_size,:],data[training_size:len(data),:1]

    return train_data, training_size, test_data, test_size


def train_test_split_multivariate(data, ratio, istest=False):

    if istest:

        # data = shuffle(data)
        training_size= int(len(data)*0.65)
        val_size = len(data)-training_size -10
        test_size = 10
        train_data,val_data,test_data = data[0:training_size,:],data[training_size:len(data)-10,:1], data[len(data)-10:,:]

        return train_data, training_size, val_data, val_size, test_data, test_size

    # data = shuffle(data)
    training_size= int(len(data)*0.65)
    test_size= len(data)-training_size
    train_data,test_data= data[0:training_size,:],data[training_size:len(data),:]

    return train_data, training_size, test_data, test_size



# convert an array of values into a dataset matrix
## setting timestep by defualt=1
def dataset_creation(data, time_step=1, ismultivariate=False):

    if ismultivariate:

        dataX, dataY = [], []
        for i in range(len(data)-time_step-1):
            a = data[i:(i+time_step), :]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(data[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    else:
        dataX, dataY = [], []
        for i in range(len(data)-time_step-1):
            a = data[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(data[i + time_step, 0])
        return np.array(dataX), np.array(dataY)


def save_json(filename,data_to_dump):
    with open(filename, 'w') as f:
        json.dump(data_to_dump, f)

