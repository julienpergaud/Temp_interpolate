#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask_restful import Api
from .api_network import AllNetwork, GetNetwork, GetNetworkByName
from .api_network_csv import AllNetworkCSV, GetNetworkCSV, GetNetworkByNameCSV
from .api_sensor import AllSensors, GetSensorsByNetwork, GetParamByNetwork
from .api_sensor_csv import AllSensorsCSV, GetSensorsByNetworkCSV, GetParamByNetworkCSV
from .api_variable import GetVariableByNetwork
from .api_variable_csv import GetVariableByNetworkCSV
from .api_measure import GetDataByNetwork, GetDataBySensor
from .api_measure_csv import GetDataByNetworkCSV, GetDataBySensorCSV

def init_api(app):
    api = Api(app)
    api.add_resource(AllSensors, '/api/sensors')
    api.add_resource(GetParamByNetwork, '/api/<netname>/sensors_param')
    api.add_resource(GetSensorsByNetwork, '/api/<netname>/sensors')
    api.add_resource(AllNetwork, '/api/networks')
    api.add_resource(GetNetwork, '/api/network/<int:netid>')
    api.add_resource(GetNetworkByName, '/api/network/<netname>')
    api.add_resource(GetVariableByNetwork, '/api/<netname>/variables') 
    api.add_resource(GetDataByNetwork, '/api/network/<netname>/measures')
    api.add_resource(GetDataBySensor, '/api/<sensor>/measures')
    
    
    api.add_resource(AllSensorsCSV, '/api/sensors/csv')
    api.add_resource(GetSensorsByNetworkCSV, '/api/<netname>/sensors/csv')
    api.add_resource(GetParamByNetworkCSV, '/api/<netname>/sensors_param/csv')
    api.add_resource(AllNetworkCSV, '/api/networks/csv')
    api.add_resource(GetNetworkCSV, '/api/network/<int:netid>/csv')
    api.add_resource(GetNetworkByNameCSV, '/api/network/<netname>/csv')
    api.add_resource(GetVariableByNetworkCSV, '/api/<netname>/variables/csv')
    api.add_resource(GetDataByNetworkCSV, '/api/network/<netname>/measures/csv')
    api.add_resource(GetDataBySensorCSV, '/api/<sensor>/measures/csv')