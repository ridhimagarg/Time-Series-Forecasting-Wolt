import utility
import multivariate_model
import univariate_model
import data_processing
import pandas as pd


data = pd.read_csv('orders_autumn_2020.csv')

data = data_processing.process_data(data)

print(data)

df_hourly_orders, X_train, y_train, X_test, ytest = univariate_model.transform_data(data)

print(X_train.shape)
print(y_train.shape)

train_predict, test_predict = univariate_model.model_structure(X_train, y_train, X_test, ytest)

print(train_predict)

univariate_model.plot_output(df_hourly_orders, train_predict, test_predict)