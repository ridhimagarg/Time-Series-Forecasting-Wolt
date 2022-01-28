import numpy as np
import pandas as pd
import json
import ast
import utility



def process_data(data):

    print("Data", data)

    print("Starting date from data: ", min(data["TIMESTAMP"]))
    print("Ending date from data: ", max(data["TIMESTAMP"]))


    data["time"] = pd.to_datetime(data["TIMESTAMP"])
    data["date"]= data["time"].dt.date
    data["hour"] = data["time"].dt.hour
    data["weekday"] = data["time"].dt.weekday
    data["dist_user_venue"] = data.apply(lambda x: utility.calculate_dist_user_venue((x["USER_LAT"], x["USER_LONG"]), (x["VENUE_LAT"], x["VENUE_LONG"])), axis=1)


    return data