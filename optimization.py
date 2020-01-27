

'''
in this module various  function handle scheduling chores
this is created to support the operation
in case some physical entity needs to be scheduled
'''

'''
##########################################################################################################################################
################################ required libraries ######################################################################################
##########################################################################################################################################
'''
import numpy as np
import pandas as pd;
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import sys
import os
import tqdm
import logging
import yaml
import requests


'''
##########################################################################################################################################
 ################################ Parameters, constants' definition ######################################################################
##########################################################################################################################################
'''


configFile = "config.yaml"
try:
    iteration_number = int(sys.argv[1])
except:
    iteration_number = 1e4

'''
##########################################################################################################################################
##########################################################################################################################################
'''


def initiale_population(time_horizon, population_num):
    '''

    creates initial so called chromosomes that form an initial genration
    this generation is created by random solutions, but solutions are feasible
    a contrint propagation method garantees that all solutions (profiles) are feasible solutions
    each solution represents the State of Charge of a battery
    :param time_horizon:
    :param population_num:
    :return:
    '''

    population_SoC = []; population_power = []

    # creation of each chromosome starts here
    for k in range(population_num):
        # create two upper and lower limits as the global contrints of the problem
        ## lower band initially is set to zero, upper band set to one
        lower_band = np.zeros(time_horizon);
        upper_band = np.ones(time_horizon);

        # following bands are temporary limits for the local constraints propagation
        down_band = np.zeros(time_horizon)
        up_band = np.ones(time_horizon);

        # initial SoC vector per solution
        SoC_array = np.zeros(time_horizon)
        # the slots to be set are selected randomly, so the distribution pattern cover all the search space even with low number of solutions
        slot_order = random.sample(range(time_horizon), time_horizon);
        # reuqired for local constraint propagations
        upper_matrix = []; down_matrix = []

        # creation of each genes starts here
        for i in slot_order:

            slot_num = i

            init_soc = np.random.uniform(down_band[i], up_band[i])

            SoC_array[slot_num] = init_soc; upper_band[slot_num] = SoC_array[slot_num]; lower_band[slot_num] = SoC_array[slot_num]

            upper_band[slot_num:] = dEmax * np.arange(0, len(SoC_array) - slot_num,1) + SoC_array[slot_num]

            upper_band[::-1][len(SoC_array)-slot_num:] = dEmax * np.arange(1, slot_num+1, 1) + SoC_array[slot_num]

            upper_band =  np.clip(upper_band, 0, 1)

            up_band = np.minimum(up_band ,upper_band)

            upper_matrix.append(up_band)


            lower_band[slot_num:] = SoC_array[slot_num] - dEmax * np.arange(0, len(SoC_array) - slot_num,1)

            lower_band[::-1][len(SoC_array)-slot_num:] = SoC_array[slot_num] - dEmax * np.arange(1, slot_num+1, 1)

            lower_band = np.clip(lower_band, 0, 1)

            down_band = np.maximum(down_band , lower_band)

            down_matrix.append(down_band)

        population_SoC.append(upper_matrix[-1])

        delta_P = np.diff(population_SoC[k])/np.diff(np.arange(0, time_horizon), 1)

        population_power.append(delta_P)

    return population_power, population_SoC


def getLastProfileUpdated(configParams, logger, filename='temp-repository/hypothesis_profile.xlsx'):
    '''
    as this optimization is called, this frunction loads the last load profiles which are subjects to optimization
    :param configParams:
    :param logger:
    :param filename:
    :return:
    '''
    try:

        #df = requests.get(configParams['http rest apis']['dsf_connectors']['loads']).json()
        df = requests.get(configParams['http rest apis']['dsf_connectors']['loads_']).json()

    except:

        logger.info('Unable to fetch data from DSF Connectors. I load profiles from local repository', extra = extraLoggerMsg)

        df = pd.read_excel(filename, sheet_name='Sheet1')[:96]

        non_flex_agg = (df['Load'].values - df['PV'].values).reshape(-1, 4).mean(axis=1)

        df_summ = pd.DataFrame(dict(aggregated_load_production = non_flex_agg, inflex_load = df['Load'].values.reshape(-1, 4).mean(axis=1),
                                   PV = df['PV'].values.reshape(-1, 4).mean(axis=1)))
    return non_flex_agg, df_summ


if __name__ == '__main__':

    # 1 ) getting the configuration Parameters
    with open(configFile, 'r') as configuration:

        try:
            configParams = yaml.safe_load(configuration)

        except yaml.YAMLError as error:

            print('Loadign model is failed. \n the configuartion file is not opened correctly or relevant fields do not match\ the error is {} \n\n retry it again manually ...'.format(error))

    method =  configParams['calculation']['optimization']['assets']['battery']['method']

    # 2) build the log file
    extraLoggerMsg = {'client': configParams['http rest apis']['dsf_connectors']['client']['entity'],
                      'IP': configParams['http rest apis']['dsf_connectors']['client']['IP'],
                      'method': method}
    formatter      = logging.Formatter('%(asctime)s {%(levelname)s}: [%(module)s][%(method)s]--> Client:[%(client)s][%(IP)s], %(message)s')

    handler = logging.FileHandler("logs/gacp_solver.log")
    handler.setFormatter(formatter)
    logger = logging.getLogger(method.upper())
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.info('Simulation is started', extra=extraLoggerMsg)

    # 3) Problem features extraction
    time_horizon = configParams['calculation']['problem']['time'] + 1
    dt =  configParams['calculation']['problem']['granularity']
    dEmax = configParams['calculation']['problem']['deltaEmax']
    kWh_max = configParams['calculation']['problem']['kWh']

    solutionNumber = configParams['calculation']['optimization'][method]['number of solutions']

    population_power, population_SoC = initiale_population(time_horizon, solutionNumber)

    # 4) Get profiles to be optimized
    non_flex_agg, df_summ = getLastProfileUpdated(configParams, logger)


    # 5) Search routine start
    SoC_derivation   = np.diff(population_SoC)
    new_profiles     = non_flex_agg + (SoC_derivation * kWh_max)
    startigPointSTD  = np.min(np.std(new_profiles.T, axis=0)),
    startigPointMEAN = np.mean(np.std(new_profiles.T, axis=0))

    population_power = np.array(population_power);
    population_SoC = np.array(population_SoC)
    error = [];
    correct_config = [];
    F_obj = np.ones(len(population_power)) * np.inf ;
    P=np.zeros(len(population_power));
    C=np.zeros(len(population_power));
    new_population = np.zeros((time_horizon) * len(population_SoC)).reshape(len(population_SoC), time_horizon)


    glob_optima_soc = None; local_optima_std = []; glob_optima_std = np.inf;  dd = 1

    t = time.time()
    glob_count = 0
    SoC_record = []


    #######################################################################################################################################################################################################
    #######################################################################################################################################################################################################
    ############# main loop starts #########################################################################################################################################################

    for glob_count in tqdm.tqdm(range(iteration_number), ncols=110, unit=' iteration',):

        SoC_derivation = np.diff(population_SoC)
        new_profiles = non_flex_agg + (SoC_derivation * kWh_max)
        local_optima_pos = np.where(np.std(new_profiles, axis=1) == min(np.std(new_profiles, axis=1)))[0][0]
        if np.std(new_profiles[local_optima_pos], axis=0) < glob_optima_std:
            glob_optima_soc = population_SoC[local_optima_pos]
            glob_optima_std = np.std(new_profiles[local_optima_pos], axis=0)
        if np.std(new_profiles[local_optima_pos], axis=0) < glob_optima_std*np.random.uniform(1,1.05):
            local_optima_std.append(min(np.std(new_profiles, axis=1)))
            SoC_record.append(population_SoC[local_optima_pos])

        F_obj = np.std(new_profiles, axis=1)
        Fitness = 1 / (F_obj + 1)
        total = np.sum(Fitness)
        P = Fitness/total
        R = np.random.uniform(0,1,len(population_SoC))

        C = np.add.accumulate(P); C = np.insert(C, 0, 0)
        for i in range(len(population_SoC)):
            for j in range(len(C)):
                if (R[i] > C[j] and R[i] < C[j+1]):
                    new_population[i] = population_SoC[j]

        population_SoC = new_population
        parents = []; parents_index = []
        ro_c = 0.1;

        while len(parents) < 2:
            R = np.random.uniform(0, 1, len(population_SoC))
            k = 0
            while (k<len(population_SoC)):
                if (R[k] < ro_c):
                    parents.append(population_SoC[k]); parents_index.append(k)
                k+=1

        Cross_points = np.random.randint(1, time_horizon-1, len(parents_index))

        k = 0; crossed_parents = [[]] * len(parents);
        temp_high_bound = np.ones(population_SoC.shape); temp_low_bound = np.zeros(population_SoC.shape);

        while k<len(parents)-1:
            crossed_parents[k] = np.concatenate((parents[k][:Cross_points[k]], parents[k+1][Cross_points[k]:]))
            # CP
            temp_high_bound[k][Cross_points[k]:] = dEmax * np.arange(0, len(parents[0]) - Cross_points[k], 1) + \
            parents[k][Cross_points[k]]
            temp_high_bound[k][::-1][len(parents[0]) - Cross_points[k]:] = dEmax * np.arange(1, Cross_points[k]+1, 1) + \
            parents[k][Cross_points[k]]

            temp_low_bound[k][Cross_points[k]:]=parents[k][Cross_points[k]]-dEmax*np.arange(0,len(parents[0])-Cross_points[k],1)
            temp_low_bound[k][::-1][len(parents[0])-Cross_points[k]:] = parents[k][Cross_points[k]] - \
            dEmax * np.arange(1, Cross_points[k]+1, 1)

            crossed_parents[k] = np.minimum(crossed_parents[k], temp_high_bound[k])
            crossed_parents[k] = np.maximum(crossed_parents[k], temp_low_bound[k])
            k+=1
        crossed_parents[k] = np.concatenate((parents[k][:Cross_points[k]], parents[0][Cross_points[k]:]))
        temp_high_bound[k][Cross_points[k]:] = dEmax * np.arange(0, len(parents[0]) - Cross_points[k], 1) + \
        parents[k][Cross_points[k]]
        temp_high_bound[k][::-1][len(parents[0]) - Cross_points[k]:] = dEmax * np.arange(1, Cross_points[k]+1, 1) + \
        parents[k][Cross_points[k]]

        temp_low_bound[k][Cross_points[k]:]=parents[k][Cross_points[k]]-dEmax*np.arange(0,len(parents[0])-Cross_points[k],1)
        temp_low_bound[k][::-1][len(parents[0])-Cross_points[k]:] = parents[k][Cross_points[k]] - \
        dEmax * np.arange(1, Cross_points[k]+1, 1)

        crossed_parents[k] = np.minimum(crossed_parents[k], temp_high_bound[k])
        crossed_parents[k] = np.maximum(crossed_parents[k], temp_low_bound[k])

        for i in range(len(parents_index)):
            population_SoC[parents_index[i]] = crossed_parents[i]

        ro_m = 0.01
        mutuated_number = int(round(population_SoC.size*ro_m))
        # here CP must come into the scene

        gens_pos_mut = np.random.randint(0, population_SoC.size, mutuated_number)
        gens_val_mut = np.random.uniform(0, dEmax, mutuated_number)

        flatten_soc = np.concatenate(population_SoC.reshape(1, population_SoC.size))
        for i in gens_pos_mut:
            if i >= len(flatten_soc)-1:
                min_lim = flatten_soc[i-1] - dEmax
                max_lim = flatten_soc[i-1] + dEmax
            elif i == 0:
                min_lim = flatten_soc[i+1] - dEmax
                max_lim = flatten_soc[i+1] + dEmax
            else:
                min_lim = max(flatten_soc[i-1], flatten_soc[i+1]) - dEmax
                max_lim = min(flatten_soc[i-1], flatten_soc[i+1]) + dEmax
            min_lim, max_lim = np.clip([min_lim, max_lim], 0, 1)
            gens_val_mut = np.random.uniform(min_lim, max_lim)
            flatten_soc[i] = gens_val_mut
        population_SoC2 = flatten_soc.reshape(population_SoC.shape)
        population_SoC = population_SoC2

        #glob_count += 1

        percentage = 1 - ((iteration_number - glob_count) / iteration_number)
        time_msg = "\rSearch Progress at {0:.2%} ".format(percentage)
        #sys.stdout.write(time_msg)
        #sys.stdout.flush()

#plt.plot(np.diff(glob_optima_soc)*kWh_max)
    if configParams['calculation']['optimization']['gacp']['plot results']:

        fig = plt.figure()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        ax = fig.add_subplot(111)
        # plots the const function evlution along the computation iterations
        #plt.plot(, 'r')
        #plt.grid()
        #plt.title('Cost function', fontsize= 22)
        ### set up the subplots position
        #gs = gridspec.GridSpec(2,2)
        #ax.set_position(gs[0:1].get_position(fig))
        #ax.set_subplotspec(gs[0:2])

        fig.add_subplot(gs[3])
        # plots the optimal operating area instructions
        plt.bar(range(len(glob_optima_soc)), glob_optima_soc, color='g')
        plt.grid()
        plt.title('optimal operating fashion', fontsize= 22)
        ### plots the aggregated profile if the instructions are totally respected
        #fig.add_subplot(gs[2])
        #plt.plot(optimal_operating_area+aggregatedProfiles)
        #plt.ylim([0,(optimal_operating_area+aggregatedProfiles).max()*1.5])
        #plt.title('resulting profile', fontsize= 22)
        ### handle fig object
        fig.tight_layout()
        plt.grid()
        plt.show()
