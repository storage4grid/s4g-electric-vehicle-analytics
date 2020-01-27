

import pandas as pd; import numpy as np; import datetime; import calendar; import math 

df_events = pd.read_csv('EVENTS',index_col=0)
df_parkings = pd.read_csv('PARKING',index_col=0)
df_plug = pd.read_csv('plugin_distribution', header = None)

FORECAST_HORIZON_DAYS=2
starting_day = datetime.datetime.now().weekday()
forecast_days = calendar.day_name[starting_day: starting_day + FORECAST_HORIZON_DAYS]

total_forecast_horizon = np.array([])
for day in forecast_days:
    num_mu, num_sigma = df_events['mu'][day], df_events['sigma'][day]
    number_of_EVs = abs(math.ceil(np.random.normal(num_mu, num_sigma, 10).mean()))

    distr_factor = 100 
    dist_list = [[i]*int(round(df_plug['pdf'][i] * distr_factor)) for i in list(df_plug.index)]
    dist_list = np.concatenate(dist_list)
    plugging_time = np.random.choice(dist_list, number_of_EVs).astype(np.int)

    durations = np.round(abs(np.random.normal(df_parkings.mu[plugging_time].values, df_parkings.sigma[plugging_time].values, number_of_EVs))).astype(np.int)

    blank_array_charge = np.zeros(288)
    unique, counts = np.unique(plugging_time, return_counts=True)
    #plugging_time[np.where(plugging_time == unique[counts>1][0])[0][0]] += 1 ### Attention: numpy doesn't consider the repetitive elements!
    for i in plugging_time:
        blank_array_charge[i] += 1
    for i in plugging_time+durations:
        blank_array_charge[i] -= 1

    blank_array_sum = (np.cumsum(blank_array_charge)*3500).repeat(5)
    total_forecast_horizon = np.append(total_forecast_horizon, blank_array_sum)

