#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask_restful import Resource, reqparse, abort
from ..models import network, sensors, sensors_param
from . import api_tools

class AllSensors(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('token')
        args = parser.parse_args()
        user = api_tools.challenge_token(args['token'])
        net = user.networkList
        if(net == None):
            abort(404, message="Networks not found.")
        
        items = []
        for n in net:
            itemSensor = sensors.query.filter_by(id_network=n).all()
        
            for i in itemSensor:
                items.append(i.to_dict())
        return items 
    
    
class GetSensorsByNetwork(Resource):
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
                    items.append(item.to_dict())
                
                return items
            else:
                abort(403, message="Access denied")   
        else:
            abort(404, message="Network not found")
            
class GetParamByNetwork(Resource):
    def get(self, netname):
        parser = reqparse.RequestParser()
        parser.add_argument('token')
        args = parser.parse_args()
        user = api_tools.challenge_token(args['token'])
        net = network.query.filter_by(name=netname).first()
        if(net != None):
            items = []
            if net.id_network in user.networkList:
                sensorsList = sensors.query.filter_by(id_network=net.id_network).all()
                for item in sensorsList:
                    params = sensors_param.query.filter_by(id_sensor = item.id_sensor).all()
                    for p in params:
                        items.append(p)
                    
                
                dic = list(dict.fromkeys(items))
                    
                final = []
                for i in dic:
                    final.append(i.to_dict())
                
                return final
            else:
                abort(403, message="Access denied")   
        else:
            abort(404, message="Network not found")