#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from flask_restful import Resource, reqparse, abort
from ..models import network, sensors, measures, measures_variable
from . import api_tools


class GetDataByNetwork(Resource):
    def get(self, netname):
        parser = reqparse.RequestParser()
        parser.add_argument('token')
        args = parser.parse_args()
        user = api_tools.challenge_token(args['token'])
        net = network.query.filter_by(name=netname).first()
        items = []
        if(net != None):
            if net.id_network in user.networkList:
                sensorsList = sensors.query.filter_by(id_network=net.id_network).all()
                for item in sensorsList:
                    data = measures.query.filter_by(id_station=item.id_sensor).all()
                    for d in data:
                        var = measures_variable.query.filter_by(id_type=d.measure_type).first()
                        
                        dic = d.to_dict()
                        dic["mesure_type info"] = var.to_dict()
                        items.append(dic)
                
                return items
            else:
                abort(403, message="Access denied")   
        else:
            abort(404, message="Network not found")
    
    
class GetDataBySensor(Resource):
    def get(self, sensor):
        parser = reqparse.RequestParser()
        parser.add_argument('token')
        args = parser.parse_args()
        user = api_tools.challenge_token(args['token'])
        sensor = sensors.query.filter_by(name=sensor).first()
        if sensor.id_network in user.networkList:
            items = []
            data = measures.query.filter_by(id_station=sensor.id_sensor).all()
            for d in data:
                var = measures_variable.query.filter_by(id_type=d.measure_type).first()
                
                dic = d.to_dict()
                dic["mesure_type info"] = var.to_dict()
                items.append(dic)      
            return items
        
        else:
            abort(403, message='Access denied')