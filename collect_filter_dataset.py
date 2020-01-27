import pandas as pd
import datetime, calendar
import numpy as np 
import math, os, requests, sys


devices = ['EDYNA-0004', 'EDYNA-0005', 'EDYNA-0006', 'EDYNA-0007', 'EDYNA-0008']
    
current_time = datetime.datetime.strptime('2019-01-15', '%Y-%m-%d')
last_date = datetime.datetime.strptime('2019-06-11', '%Y-%m-%d')

while current_time < last_date:
    
    time_msg = "\rExecution at {} ".format(current_time)
    sys.stdout.write(time_msg)
    sys.stdout.flush()

    start_date_dt = current_time; start_time = '00:00:00'
    end_date_dt = start_date_dt + datetime.timedelta(days=1); end_time = '00:00:00'

    start_date_str = start_date_dt.strftime('%Y-%m-%d')
    end_date_str = end_date_dt.strftime('%Y-%m-%d')
    
    time_axis = np.array([0]);  
    
    for i in range(len(devices)):
        try:
            locals()[devices[i].replace('-', '')]
        except:
            pass
        else:
            del locals()[devices[i].replace('-', '')]

        
    for device in devices:
        df_temp = []; 
        try:

            url = 'http://10.8.0.50:8086/query?db=S4G-DWH-USM&q=select%20*%20from%20%22S4G-GW-'+ device +'%22%20where%20time%20%3E%20%27'+ start_date_str +'%20'+ start_time +'%27%20AND%20time%20%3C%20%27'+ end_date_str +'%20'+ end_time +'%27'
            auth = session.post(url)
            response = session.get(url)

            df_temp = pd.DataFrame(response.json()['results'][0]['series'][0]['values'], columns=response.json()['results'][0]['series'][0]['columns'])

            exec("%s = %s" % (device.replace('-',''), list(df_temp['P'].values)))

        except:
            logger = logging.getLogger('record_empty_responses')
            logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler('No Replies.log')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
            logger.info("Empty response for period {} and device {} ".format(start_date_str, device))

        
        if len(time_axis) < len(df_temp):
            try:
                time_axis = np.array([datetime.datetime.strptime(df_temp['time'][i][:df_temp['time'][i].find('.')].replace('T', ' '), '%Y-%m-%d %H:%M:%S') for i in range(len(df_temp))])
            
            except:
                logger.info("Problem regarding time for {} and device {} ".format(start_date_str, device))
    
    temp_arrays = []  
    uniformed_matrix = np.array([])
    
    try:

        for i in range(len(devices)):
            temp_arrays.append(locals()[devices[i].replace('-', '')])

        uniformed_matrix = np.ones((np.max([len(ps) for ps in temp_arrays]), len(temp_arrays))) * 0 

        for i,c in enumerate(temp_arrays): 
            uniformed_matrix[:len(c),i]=c

        filename = start_date_str

        if  uniformed_matrix.max() < 1e3:
            filename = filename + ' USELESS!'
            logger = logging.getLogger('spam_application')
            logger.setLevel(logging.DEBUG)

            fh = logging.FileHandler('TRACKS.log')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
            logger.info("Data for the interval {} is useless with the maximum power of {} and the total length of {} ".format(start_date_str, uniformed_matrix.max(), len(time_axis)))


        df = pd.DataFrame(uniformed_matrix, index=time_axis, columns=[devices])
        df.to_csv('EDYNA_COMMERCIAL/'+ filename, sep='\t')
    except:
        pass

    current_time = current_time + datetime.timedelta(days=1)


print('\nDONE!')