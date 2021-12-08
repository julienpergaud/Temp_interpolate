#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from flask_restful import Resource, reqparse, abort
from ..models import network, sensors
from . import api_tools


class GetDataByNetworkCSV(Resource):
    def get(self, netname):
        parser = reqparse.RequestParser()
        parser.add_argument('token')
        args = parser.parse_args()
        user = api_tools.challenge_token(args['token'])
        net = network.query.filter_by(name=netname).first()

        if(net != None):
            if net.id_network in user.networkList:
                sensorsList = sensors.query.filter_by(id_network=net.id_network).all()
                query = "SELECT * FROM measures INNER JOIN measures_variable ON measures.measure_type = measures_variable.id_type WHERE measures.id_station = {} ".format(sensorsList[0].id_sensor)
                for item in sensorsList[1:]:
                    query += "OR measures.id_station = {} ".format(item.id_sensor)
                        
                return api_tools.download_file(query)
            else:
                abort(403, message="Access denied")   
        else:
            abort(404, message="Network not found")
    
    
class GetDataBySensorCSV(Resource):
    def get(self, sensor):
        parser = reqparse.RequestParser()
        parser.add_argument('token')
        args = parser.parse_args()
        user = api_tools.challenge_token(args['token'])
        sensor = sensors.query.filter_by(name=sensor).first()
        if sensor.id_network in user.networkList:
            
            query = "SELECT * FROM measures INNER JOIN measures_variable ON measures.measure_type = measures_variable.id_type WHERE measures.id_station = {}".format(sensor.id_sensor)
             
            return api_tools.download_file(query)
        
        else:
            abort(403, message='Access denied')