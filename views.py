'''
Server side

'''

from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
import time
from django.views.generic import View
from .import calculation
from .import auxCallsHandler
import logging

global extraLoggerMsg
global logger

## TODO: to set approprrieùate measure for the caller
extraLoggerMsg = {'client': 'UNKNOWN',
                  'IP': 'localhost',
                  'method': 'SA'}
formatter      = logging.Formatter('%(asctime)s {%(levelname)s}: [%(module)s][%(method)s]--> Client:[%(client)s][%(IP)s], %(message)s')

handler = logging.FileHandler("instructions/logs/server.log")
handler.setFormatter(formatter)
logger = logging.getLogger('SERVER')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

# optimal operating area calculation
@api_view(['POST'])
def optimalOperations(request):
    '''
    client requests for the optimal operating area instructions
    :param request: client request
    :return: a json containig optima operating area instruction, computation time, errors adn standard deviation as the indicators of computation
    '''
    try:
        # parse the content of the request
        requestBody=json.loads(request.body)
        # get the total predicted energy to vbe retreived by the EV charging
        totalEnergyForEVs = requestBody['expectedEVChargingEnergy']
        # number of epochs for optimization calculation
        epochs = requestBody['epochs']
        # call the main computation function and gets results and various calculation indicators
        optimal_operating_area, energyConstraintError, resultingProfileDeviation, rmseResultingProfile, elapsedCalculationTime = calculation.computeOptimalOperatingFashion(totalEnergyForEVs, epochs)
        # making the response as dictionary
        oofResponse = {"OOF":optimal_operating_area.tolist(),
                       "Error":energyConstraintError,
                       "result_std":resultingProfileDeviation,
                       "rmseTotal": rmseResultingProfile,
                       "computation_time": elapsedCalculationTime}
        # convert the dictionary to the json
        optimalOperatingAreaJsonResponse = json.dumps(oofResponse)
        logger.info('Request is received ...', extra=extraLoggerMsg)
        # deprecated: used to read data from the updated file
        if 0:
            with open('./oof.json') as jsonfile:
                setpoints = json.load(jsonfile)
        # return the response if no error raised
        return JsonResponse(optimalOperatingAreaJsonResponse, safe=False)
        # classic handeling of the errors
    except ValueError as error:
        logger.error('Server could not proceed the request because of this value error: {}'.format(error), extra=extraLoggerMsg)
        # return standard error message
        return Response(error.args[0],status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def getInstruction():
    return JsonResponse({'VALUE':0},safe=False)
