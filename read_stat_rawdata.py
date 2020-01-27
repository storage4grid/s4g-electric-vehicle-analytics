

# Copyright 2017 The LINKS. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Current module containts auxiliary functions 
for the data preprocessing class
"""


import glob, os, inspect
from os.path import expanduser, join
from pandas import *
import datetime as dt, calendar




global currentDirPath
global static_file
global LOCATION
global logger

currentDirPath = os.getcwd()
static_file    = ''
LOCATION       = ''



def get_logger():

    extra = {'function':func}

    logger = logging.getLogger('eva_static_parsers')

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s: %(function)s')

    fileHandler = logging.FileHandler('weatherDatasetLocations.log')

    fileHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)

    logger = logging.LoggerAdapter(logger, extra)

    logger.info(LOCATION)


def get_raw_data():
    '''
    reads the data from static repository and filters it
    '''
    
    extraMsg = {'function': inspect.stack()[0][3]}
    
    for file in glob.glob('localRepsitory/*.xlsx'):    
        xls = ExcelFile(file)
        df = xls.parse(xls.sheet_names[0])                
        
        stations = df.get('Indirizzo')
        list_of_stations = []
        for i in stations:
            if i not in list_of_stations:
                list_of_stations.append(i)  
                
        file_name = file.replace('.xlsx','')
        newpath = path +'\\'+ file_name
        if not os.path.exists(newpath):
            os.makedirs(newpath)   
        os.chdir(newpath)
        
        writer = ExcelWriter(file)
        for key in list_of_stations:  
            if '*' in key:
                key = key.replace('*', '')
            new_df = df[df['Indirizzo'].str.contains(key)] 
            key = key.replace('/','').replace('ü','u').replace(',','')
            new_df.to_csv(key +'.txt', sep='\t', encoding='utf-8', index=False)
            new_df.to_excel(writer,key)
        writer.save()
        

def get_per_station():
    
    extraMsg = {'function': inspect.stack()[0][3]}

    out_data = {}
    os.chdir(currentDirPath)
    number =0
    for file in glob.glob('*.xlsx'):    
        xls = ExcelFile(file)
        for sheet_ in xls.sheet_names:
            number+=1
            df = xls.parse(sheet_) 
            if sheet_ != 'Edyna': # there is just one non meaningful charging data at the moment 
                str_datetimes = [j.replace('/','-') for j in df['Inizio data CU']]
                total_horizon_datetime = [dt.datetime.strptime(i, '%d-%m-%Y %H:%M:%S') for i in str_datetimes]
                total_horizon_time = [k.time() for k in total_horizon_datetime]
                week_day = [calendar.day_name[l.weekday()] for l in total_horizon_time]
                out_data[total_horizon_time] = df['kWh']
                alendar.day_name[l.weekday()] for l in my_datetime
                
                

def get_weekdays():

    extraMsg = {'function': inspect.stack()[0][3]}
    
    df = pd.read_excel(static_file, sheet_name=None)
    durata_in_hour = []
    try:
        for i in df['Durata']:
            if 'd' in i:
                d2s = int(i[0])*24*3600
            else: d2s=0

            h2s_index = (i.find('h'))
            if h2s_index==1:
                h2s=int(i[h2s_index-1])*3600
            else: h2s=int(i[h2s_index-2:h2s_index])*3600

            m2s_index = (i.find('m'))
            if m2s_index==1:
                m2s=int(i[m2s_index-1])*60
            else: m2s=int(i[m2s_index-2:m2s_index])*60

            sec_index = (i.find('s'))
            if sec_index==1:
                sec=int(i[sec_index-1])
            else: sec=int(i[sec_index-2:sec_index])

            tot_sec = d2s+h2s+m2s+sec
            durata_in_hour.append(tot_sec/3600)
    except Exception as error:
        logger.log(error, extraMsg)


def get_chEvents_per_weekdays():

    extraMsg = {'function': inspect.stack()[0][3]}

    if not os.path.exists(LOCATION):
        os.makedirs(LOCATION)

    os.chdir(LOCATION)

    str_datetimes = [j.replace('/','-') for j in df['Inizio data CU']]
    my_datetime = [dt.datetime.strptime(i, '%d-%m-%Y %H:%M:%S') for i in str_datetimes]
    my_time = [k.time() for k in my_datetime]
    week_day = [calendar.day_name[l.weekday()] for l in my_datetime]

    filter_ = 4 ### the begining od the sampling has just 4 charging that has non meaning 
    dts = [d.date() for d in my_datetime[filter_:]]
    dtdf = pd.DataFrame(my_datetime[filter_:], columns=['datetime'])
    dtdf['date'] = dts
    dtdf=dtdf.groupby( ['date'] ).size().to_frame(name = 'count').reset_index()
    w = [calendar.day_name[i.weekday()] for i in dtdf['date']]
    dtdf['weekday'] = w

    wdf = {i:list(dtdf['count'][dtdf['weekday']==i].values) for i in list(calendar.day_name)}

    means = [np.mean(wdf[i]) for i in wdf]
    stds = [np.std(wdf[i]) for i in wdf]
    Index = list(wdf.keys())

    wdf_stat = {'means':means, 'st_dev':stds}
    wdf_stat = pd.DataFrame(wdf_stat)
    wdf_stat.index = Index
    wdf_stat.to_csv(LOCATION+'\WEEK_DAYS.txt', header=None)




                