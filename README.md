# Data Science Summer Intern assignment 2022

**Assignment for candidates**

## Table of Contents

1. [Data](#data)
3. [Task - Predicting how many orders Wolt may get in next hour](#task)
4. [Data Analysis and Modelling](#data-analysis-and-modelling)
5. [Working with files](#working-with-files)

----

## Data

* **Time series.** I have choosen this dataset [provided file](orders_autumn_2020.csv) as a process fluctuating in time


---

## Task

* **Forecast No. of orders.** - Building a forecasting model for predicting how many orders WOLT may get in next hour? 

---

## Data Analysis and Modelling

### Data Exploration :chart_with_upwards_trend:

For detail Analysis, go check this [notebook](Analysis.ipynb #Hourly).

Here are same basic insights of data -:

1. Hourly Analysis :hourglass:

![Hourly Analysis](images/hourlyanalysis.png)

2. Weekly Analysis

![Weekly Analysis](images/weekdayorders.png)

3. Routing Analysis

![Routing Analysis](images/uservenuredistance.png)


### Data Processing







### Modelling :rocket:

I have choosen LSTM for building forecasting model for predicting the no. of orders in next hour...

##### Reason of choosing LSTMN

- We are working on timeseries data and in that case we need to keep useful information from previous data and LSTMN has a memory cell which helps in keeping past information also.

- Two models I have built -> Univariate and Multivariate 

- Features for multivariate model -> No. of orders, Weather data(Wind, Precipitation, Cloud Coverage, Temperature) as the no. of orders may depend upon the weather and route[I didnt take into consideration..but can be taken]

##### Evaluation

This the output from the two models -:

1. Univariate

![Univariate Output](images/output_univariate.png)

![Loss functio](images/loss_univariate.png)

2. Multivariate

![Multivariate Output](images/output_multivariate.png)

![Loss functio](images/loss_multivariate.png)


### Further development

I have trained two models but seems like univariate is outperforming as the validation loss is better than training loss.

If more time will be there these things could be done -:

    - Missing data for weather can be handled in more efficient way
    - Different model architecture should be tried and thier metrics scores
    - Due to less data, not able to test out on different whole dataset as in the current model, information leakage happened.



### Working with files


---

## Your background and Wolt
After the practical work, let's discuss what you have learned and what your ambitions are. Write a bit about the problems you like to work with. Have you written your thesis or a larger piece of coursework about something that you would see beneficial for Wolt? If you already have work history, are there some things that you would like to try here? Based on your knowledge about us, are there some problems you would like to help us solve? Do you have some relevant, interesting minors or side projects? We are always interested in enthusiastic people with fresh ideas, and this could be the opportunity to put something you recently learned into use!
