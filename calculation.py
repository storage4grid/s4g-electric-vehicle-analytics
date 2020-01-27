'''
###############################################################################################################################################################################
Two simple optimization algorithm are provided, one is quadratic problem and second one is a simulated anealing to solving to find the optimal operating area
the inputs are:
    - Energy prices (dynamic)
    - Aggregated load profile in the area of operation (feeder or substation)
    - Total power generated from renewables
###############################################################################################################################################################################
'''


'''
###############################################################################################################################################################################
modules, constants, arguments
###############################################################################################################################################################################
'''
import numpy as np
import json
import datetime
import requests
from scipy.optimize import minimize
import yaml
import logging
import sys
import os
import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


global epochs
global evEnergy

try:
    epochs = int(sys.argv[1])
except:
    epochs = 0

try:
    evEnergy = int(sys.argv[2])
except:
    evEnergy = 0

'''
###############################################################################################################################################################################
auxiliary functions
###############################################################################################################################################################################
'''

def getLastProfileUpdated(configParams, logger, evEnergy, filename='instructions/temp-repository/predictions.json'):
    '''
    retreive the last preditions of the power consumption as well as the total energy forescast to be used by the EVs
    :param configParams:
    :param logger:
    :param evEnergy:
    :param filename:
    :return:
    '''
    # get
    try:
        I_AM_AN_INTENTIONAL_ERROR # --> to avoid waste of time in every execution, since this connectors are in vpn and the suthentication token cannot be loaded online
        aggregatedProfiles = np.array([float(i['Loads']) for i in requests.get(configParams['http rest apis']['dsf_connectors']['loads']).json()])

    except Exception as error:

        logger.warning('unable to retreive data from dsf connectors because: {}'.format(error), extra = extra)
        # read the data from last saved file
        with open(filename) as json_file:

            data = json.load(json_file)

        # last update of the aggregted power in the substation/feeder of interest
        aggregatedProfiles = np.array(data['aggregatedProfile'])

    # total energy to be consumed by EVs
    if not evEnergy:
        # if not given by the command line arg, get it from json file
        evEnergy = data['evEnergy']

    # the total horizon of the forecast
    computationHorizon = aggregatedProfiles.shape[0]

    return aggregatedProfiles, computationHorizon, evEnergy


def getEnergyPrices(configParams, logger):
    '''
    gets the energy price either from 3rd party or file or ...
    :param configParams: configuration dict
    :param logger: logger object
    :return: price vector
    '''
    try:
        ## TODO: for development purposes
        I_AM_AN_INTENTIONAL_ERROR # --> to avoid waste of time in every execution, since this connectors are in vpn and the suthentication token cannot be loaded online
        prices_connector = requests.get(configParams['http rest apis']['dsf_connectors']['prices'])
        # extract json file
        prices = prices_connector.json()['Publication_MarketDocument']['TimeSeries'][0]['Period']['Point']
        # extract the interested fields
        prices_array_parse = np.array([position['price.amount'] for position in prices])
        # convert the vector to numpy floating points
        prices_array = prices_array_parse.astype(np.float)
    # it falls of course here
    except Exception as error:
        # and reads the fixed array
        prices_array = np.array([36, 34, 33, 32, 32, 36, 46, 53, 59, 57, 51, 48, 46, 46, 48, 50, 50, 56, 57, 54, 49, 46, 44, 40])
        logger.warning('Unable to retreive data from connectors: {}'.format(error), extra = extra)

    return prices_array


def getConfigs(logger, configFile = "instructions/config.yaml"):
    '''
    reads the configuration file
    :param logger: logger object
    :param configFile: configuration file name
    :return: configuration parameters as dict
    '''
    with open(configFile, 'r') as configuration:

        try:
            configParams = yaml.safe_load(configuration)

        except yaml.YAMLError as exc:
            logger.debug(exc, extra=extra)

    return configParams

def getStartingStateSA(horizon):
    '''
    initialization of the solving order (deprecated)
    :param horizon:
    :return:
    '''

    """ gives a random order of solveing problem """

    np.random.seed(0)

    assignmentOrders = np.random.choice(range(computationHorizon), computationHorizon, replace=False)

    return assignmentOrders


def createLoggerObject():
    '''
    creates/get the logger object
    :return: logger object
    '''
    # extra standard fields of the logger message
    global extra
    ## TODO: it should get the client IP
    extra = {'client': 'unknown'}
    # created instance of the logger object
    logger = logging.getLogger('OOF')
    logger.setLevel(logging.INFO)
    # formatter object
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s, | %(client)s')
    # add file handler
    fileHandler = logging.FileHandler('instructions/logs/oof instructions.log')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger = logging.LoggerAdapter(logger, extra)

    return logger

def getTimingSlots():
    '''
    builds a table of timig slots for case of partially-dynamic prices, that means variable pricing slots
    :return: timing slots
    '''
    # f1, f2 and f3 represent various timing slots of day in which the energy price is defined differently
    f3 = np.arange(0,6,1); f3 = np.append(f3, 23)
    f2 = np.arange(6,7,1); f2 = np.append(f2, np.arange(19,23,1))
    f1 = np.arange(7,19,1)

    return f1, f2, f3


def solveQp(pm, prices_array, t1, t2, t3, configParams):
    '''
    solve optimization problem as a quadratic problem
    :param pm: total energy to be distributed
    :param prices_array: energy price vector
    :param t1: timing slot 1
    :param t2: timing slot 2
    :param t3: timing slot 3
    :param configParams: parameters of optimization problem
    :return:
    '''
    """ it solves simplty a sequential quadratic problem, which suits this timeseries problem """
    total_forecast_charging = pm
    # definition of the objective problem
    def objective(x):
        f = np.sum(x*prices_array)
        return f
    # ktt inequality constraints
    def ineqQonstraint(x):
        return np.sum(x) - x
    # ktt equality constrint (satisfaction of energy)
    def eqConstraint(x):
        return total_forecast_charging-np.sum(x)
    #number of steps
    n = 24
    #initial guess
    x0 = np.random.uniform(-1,n+1,n)
    # up and down bounds
    b = (0.0, configParams['physical_system']['grid']['max_power'])
    bnds = tuple(list(b for i in range(n)))
    # set up constraints according to scipy convention
    con1 = {'type': 'ineq', 'fun': ineqQonstraint}
    con2 = {'type': 'eq', 'fun': eqConstraint}
    cons = ([con1,con2])
    # selection of the optimization solver and call the relevant function
    solution = minimize(objective,x0,method='SLSQP',\
                        bounds=bnds,constraints=cons)
    # gets the results
    y = x = solution.x
    # prunes the resultsing arrays
    if prices_array.shape[0] < 1:

        results = np.zeros(24)
        f1_timing = sum(x[f1])/len(f1)
        results[f1] = f1_timing

        f2_timing = sum(x[f2])/len(f2)
        results[f2] = f2_timing

        f3_timing = sum(x[f3])/len(f3)
        results[f3] = f3_timing

    return results, x0

def solveSa(configurationParams, profile, T, evEnergy, epochs):
    '''
    Simulated Anealing (SA) is an optimzation function used to calculate the optimal operating area. The calculation procedure fits the required energy as if it was totally flexible.
    :param conf: configuration parameters
    :param profile: the subject profile (unflexible) to be improved via available resources
    :param T: total horizon of the computation
    :param evEnergy: total energy predicted EVs require within T time horizon
    :param epochs: number of global computation epochs
    :return: optimal charging profile for the EVs (V2G is assumed to be possible)
    '''
    # constrints of the electricity system
    lowerBoundPower = configurationParams['calculation']['optimal operatin area']['sa']['lowerBoundPower']
    higherBoundPower = configurationParams['calculation']['optimal operatin area']['sa']['higherBoundPower']
    # this parameters allows to search wide or in limited space. It is set to constnt. It is interesting to vary it gradually with anealing temperature
    globalTemperture = configurationParams['calculation']['optimal operatin area']['sa']['globalTemperture']
    # it is important to set objective function
    # for sake of simplicity and development purposes, this objective function is set to "load leveling"
    # It can be:
    #   - to reduce total energy price spent for the community
    #   - to shave the peak of power
    #   - to increase usage of local renewables
    #   - AND MORE as DESIRED...
    objectiveProfile = np.repeat((profile+evEnergy/T).mean(), T)
    # keep track of the errors in optimization process
    costs_record = []
    costs_record_global = []
    # keep track of the temperature of anealing
    temperatureTrack = []
    # initialize the optimal configuration vector
    optimalConfiPerEpoch = np.zeros(T)
    # gets the initial anealing temperature
    anealingTemperature     = configurationParams['calculation']['optimal operatin area']['sa']['anealingTemperature']
    # gets the constant of the first order equation that handles the temperature decrease
    alpha     = configurationParams['calculation']['optimal operatin area']['sa']['alpha']
    beta     = configurationParams['calculation']['optimal operatin area']['sa']['beta']
    # simply handles the number of epochs if it is not given by the command line command
    if not epochs:
        epochs = configurationParams['calculation']['optimal operatin area']['sa']['epochs']
    # number of internal iteration per epoch
    local_search_iterations = configurationParams['calculation']['optimal operatin area']['sa']['local search iterations']
    # initialize the cost function
    costFuncLocalOptima  = np.inf
    # initialize the optimal charging profile
    evChargingProfile       = np.repeat(evEnergy/T, T) # objective variables
    # starts the global search
    for epoch in tqdm.tqdm(range(epochs)):
        # starts to deacrease temperature per epoch according to a second order equation
        anealingTemperature -= alpha * anealingTemperature + beta
        # makes sure temperature doesn't go under zero if the optimization parameters are was not carefully set
        anealingTemperature = np.clip(anealingTemperature, 0.1, np.inf)
        # for debug reason
        temperatureTrack.append(anealingTemperature)
        # internal search starts
        for iteration in range(local_search_iterations):
            # at each iteration the search per step order is randomly selected
            assignmentOrders = np.random.choice(range(T), T, replace=False)
            # search for each step in optimization horizon
            for index in assignmentOrders:
                # mean value per step
                temporaryLocalMeanValue    = evChargingProfile.sum() - evEnergy
                # standard deviation per step
                temporaryLocalStdDeviation =  higherBoundPower # globalTemperture #* (1-(epoch/epochs))
                # energy per step
                temporaryDtEnergy          = np.random.normal(loc=temporaryLocalMeanValue,
                                                              scale=temporaryLocalStdDeviation,
                                                              size=1)
                # makes sure the setup power (energy) is not out of physical system's bounds
                temporaryPowerPerIndex     = np.clip(temporaryDtEnergy, lowerBoundPower, higherBoundPower)
                # last accepted solution
                currentSolution = np.copy(evChargingProfile)
                # penalty term regarding total energy constraints
                currentEnergySatisConstraint  = np.square(abs(currentSolution.sum() - evEnergy))
                # cost only for objective function
                currentLocalobjectiveFunction = np.square(np.subtract(currentSolution+profile, objectiveProfile)).mean()

                nextSolution    = np.copy(currentSolution)
                # discard the solution with discharge if V2G isn't set
                if configurationParams['calculation']['optimal operatin area']['sa']['V2G']:
                    nextSolution[index]  = np.clip(temporaryPowerPerIndex, lowerBoundPower, higherBoundPower)
                else:
                    nextSolution[index]  = np.clip(temporaryPowerPerIndex, 0, higherBoundPower)
                # panalty term for energy constraint of the new solution
                nextEnergySatisConstraint  = np.square(abs(nextSolution+profile.sum() - evEnergy))
                # cost only for objective function of new solution
                nextLocalobjectiveFunction = np.square(np.subtract(nextSolution+profile, objectiveProfile)).mean()
                # total cost for last accepted solution
                currentLocalObjectiveTotalFunction = currentLocalobjectiveFunction + currentEnergySatisConstraint
                # improvment in cost function
                deltaCostFunction =  nextLocalobjectiveFunction - currentLocalobjectiveFunction
                # if cost is reducedd, it is accepted imediately
                if deltaCostFunction < 0:
                    evChargingProfile = np.copy(nextSolution)
                    costs_record_global.append(nextLocalobjectiveFunction)
                # otherwise it should be undergo for:
                else:
                    # if anealing temperature is still high and/or the worsening of the cost function was not that high, there will be another chance
                    if np.exp(-deltaCostFunction/anealingTemperature) > np.random.uniform(size=1):
                        evChargingProfile = np.copy(nextSolution)
                        costs_record_global.append(nextLocalobjectiveFunction)

    return evChargingProfile, costs_record_global


'''
______________________________________ getting optimal profile_______________________________________________________________________________________________________________________________________
the problem can be solve to:
    - maximize usage of the RES
    - minimize total energy cost
    - load levelling
    - load shifting
_____________________________________________________________________________________________________________________________________________________________________________________________________
'''

#if __name__ == '__main__':
def computeOptimalOperatingFashion(evEnergy, epochs):

    print('script starts to fetch required data...')
    # gettign the logger object
    logger = createLoggerObject()
    # getting and parsing the configuration parameters
    configParams = getConfigs(logger)
    # getting the calculation variables from various resources
    aggregatedProfiles, computationHorizon, evEnergy = getLastProfileUpdated(configParams, logger, evEnergy)
    # getting the prices from 3rd
    prices_array = getEnergyPrices(configParams, logger)
    # getting the time slot based energy prices, based on the calculation subject
    t1, t2, t3 = getTimingSlots()
    # starts to calculate the optimization problem
    try:
        tic=time.time()
        optimal_operating_area, errorTrack = globals()['solve'+ configParams['calculation']['optimal operatin area']['optimization']['method'].capitalize()](configParams, aggregatedProfiles, computationHorizon, evEnergy, epochs)
        toc=time.time()
        # extracts some indicative values and results
        ## shows how much energy constraint (difference between reference energy and the one instructed) is out
        energyConstraintError     = 100*abs((optimal_operating_area.sum()-evEnergy)/optimal_operating_area.sum())
        ## calculates the standard deviation for the resulting profile after adding up the optimal charging profile of the EVs
        resultingProfileDeviation = (optimal_operating_area+aggregatedProfiles).std(axis=0)
        ## objective was:
        objectiveProfile = np.repeat((optimal_operating_area.sum(axis=0) + aggregatedProfiles.sum(axis=0) + evEnergy)/computationHorizon, computationHorizon)
        ## calculate the root mean square error for the resultign profile
        rmseResultingProfile      = np.sqrt(np.square(np.subtract(optimal_operating_area+aggregatedProfiles, objectiveProfile)).mean())
        print(rmseResultingProfile)
        ## elapsed computation time
        elapsedCalculationTime    = toc-tic
        print('\nCalculation of Optimal Operatin Area is calculated successfully within {} seconds.'.format(toc-tic))
        # loogs the results
        logger.info('Optimal profile is computed successfully. Energy satisfaction error in {} steps is {} %, standard deviation of {} with R.M.S.E. equal to {} in {} epochs that took {} seconds.'.format(computationHorizon,
        energyConstraintError, resultingProfileDeviation, rmseResultingProfile , epochs, elapsedCalculationTime), extra=extra)

    except Exception as error:
        print('Calculation is failed to complete!:\n {}'.format(error))

    # the results of the optimization problem which is Optimal Operatin Fashion/Area is written on json
    print('writing the results ...')
    with open('./oof.json', 'w') as f:
        json.dump({'optimalChargingProfile':optimal_operating_area.tolist()}, f)
    # the process is finished
    print('\nthe results are comunicated.')
    # if the plotting option is enabled it gives back a simple overview of the results and errors evolutions
    if configParams['calculation']['optimal operatin area']['optimization']['plot results']:
        fig = plt.figure()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        ax = fig.add_subplot(111)
        # plots the const function evlution along the computation iterations
        plt.plot(errorTrack, 'r')
        plt.grid()
        plt.title('Cost function', fontsize= 22)
        # set up the subplots position
        gs = gridspec.GridSpec(2,2)
        ax.set_position(gs[0:1].get_position(fig))
        ax.set_subplotspec(gs[0:2])

        fig.add_subplot(gs[3])
        # plots the optimal operating area instructions
        plt.bar(range(len(optimal_operating_area)), optimal_operating_area, color='g')
        plt.grid()
        plt.title('optimal operating fashion', fontsize= 22)
        # plots the aggregated profile if the instructions are totally respected
        fig.add_subplot(gs[2])
        plt.plot(optimal_operating_area+aggregatedProfiles)
        plt.ylim([0,(optimal_operating_area+aggregatedProfiles).max()*1.5])
        plt.title('resulting profile', fontsize= 22)
        # handle fig object
        fig.tight_layout()
        plt.grid()
        plt.show()
    time.sleep(0.2)
    print('\nClosing session ...')

    return optimal_operating_area, energyConstraintError, resultingProfileDeviation, rmseResultingProfile, elapsedCalculationTime

computeOptimalOperatingFashion(evEnergy, epochs)
