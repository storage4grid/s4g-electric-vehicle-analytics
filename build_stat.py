import pandas as pd; import numpy as np; import os, sys, glob; import matplotlib.pyplot as plt; import datetime; import calendar; import math 

os.chdir('\dataset')

print('STARTS TO GET STATISTICS OUT OF THE CLEANED AND ELABORATED DATA')
# STEP to REMOVE nan VALUES
events_number = pd.DataFrame({x:dict(mu=0, sigma=0) for x in weekdays_lists}).T   # dict(mu=[], sigma=[]); 
plugging_time = dict(mu=[], sigma=[]); parking_duration = dict(mu=[], sigma=[])
total_pluggins = np.array([])
correlated_plugging, correlated_parking = np.array([]), np.array([])

for day_ in weekdays_lists:
    locals()[day_ + '_DICT'] = dict(); PLUG_TIMES = np.array([]); DURATIONS = np.array([])
    PLUG_TIMES = np.array([]); DURATIONS = np.array([]); EVENTS = np.array([])
    for example in range(locals()[day_ + '_final_shape'].shape[0]):
        for meter in range(locals()[day_ + '_final_shape'].shape[2]):
            for sample in range(locals()[day_ + '_final_shape'].shape[1]):
                if np.isnan(locals()[day_ + '_final_shape'][example][:,meter][sample]):
                    try:
                        locals()[day_ + '_final_shape'][example][:,meter][sample] = locals()[day_ + '_final_shape'][example][:,meter][sample-1]
                    except:
                        locals()[day_ + '_final_shape'][example][:,meter][sample] = 0

        # DETECTS ALL EVENTS, AND THEN DISAGGREAGATES THEM BASED ON CHARGING AND PLUGING OUT.
        events = np.vstack((np.diff(locals()[day_ + '_final_shape'][example][:,:], axis=0), np.zeros(locals()[day_ + '_final_shape'].shape[2])))                
        events_indices = np.concatenate([np.nonzero(np.r_[1, events[:,kappa][:-1]])[0][1:] - 1 for kappa in range(locals()[day_ + '_final_shape'].shape[2])])

        # to test because the weekend could be empty of data 
        try:
            plugging_time = events_indices[::2]
            charge_duration = events_indices[1::2] - plugging_time
            PLUG_TIMES = np.append(PLUG_TIMES, plugging_time)
            DURATIONS = np.append(DURATIONS, charge_duration)
            EVENTS = np.append(EVENTS, len(plugging_time))
        except:
            print('ATTENTION to sample {} !'.format(sample))
        
        if len(plugging_time) == len(charge_duration):
            correlated_plugging, correlated_parking = np.append(correlated_plugging, plugging_time), np.append(correlated_parking, charge_duration)



    locals()[day_ + '_DICT']['PLUG_TIMES'] = PLUG_TIMES
    locals()[day_ + '_DICT']['EVENTS'] = EVENTS
    locals()[day_ + '_DICT']['DURATIONS'] = DURATIONS

    if len(PLUG_TIMES) == len(DURATIONS): print('operation was correct.')
    else: print('there are inconsistency in data.')

    try:
        events_number['mu'][day_] = abs(math.ceil(np.random.normal(np.mean(locals()[day_ + '_DICT']['EVENTS']))))
        events_number['sigma'][day_] = abs(math.ceil(np.random.normal(np.std(locals()[day_ + '_DICT']['EVENTS']))))
        total_pluggins = np.append(total_pluggins, PLUG_TIMES)
    except:
        pass
    
    
#### ARRIVING (PLUGGING) TIME DISTRIBUTION vs. TIME      
blank_array = np.zeros(288).astype(np.int)
unique, counts = np.unique(total_pluggins.astype(np.int), return_counts=True)
blank_array[np.array(list(zip(unique, counts)))[:,0]] =  np.array(list(zip(unique, counts)))[:,1]
plugin_distribution = blank_array * 100 /blank_array.sum()

#### PARKING STATISTICAL MODEL BASED ON ARRIVAL TIME
batchsize = 5
dict_parkings = dict(mu=np.array([]), sigma=np.array([]))
for i in range(0, len(blank_array)): 
    batch_array = np.where((correlated_plugging>=i-2) & (correlated_plugging<i+3))
    mean_values = correlated_parking[batch_array[0]].mean()
    std_values = correlated_parking[batch_array[0]].std()
    dict_parkings['mu'] = np.append(dict_parkings['mu'], mean_values)
    dict_parkings['sigma'] = np.append(dict_parkings['sigma'], std_values)
    
df_parkings=pd.DataFrame(dict_parkings)

mu_nans, sigma_nans = df_parkings[df_parkings.notnull()].mean()
df_parkings.mu[df_parkings.mu.isna()], df_parkings.sigma[df_parkings.sigma.isna()] = mu_nans, sigma_nans

events_number.to_csv('EVENTS')
df_parkings.to_csv('PARKING')
np.savetxt('plugin_distribution', plugin_distribution)