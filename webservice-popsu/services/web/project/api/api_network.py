#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from flask_restful import Resource, reqparse, abort
from ..models import network
from . import api_tools

class AllNetwork(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('token')
        args = parser.parse_args()
        user = api_tools.challenge_token(args['token'])
        net = user.networkList
        if(net == None):
            abort(404, message="Networks not found")     


        items = []
        
        for n in net:
            item = network.query.filter_by(id_network=n).first()
            items.append(item.to_dict())
        
        if len(items) == 0:
            abort(404, message="Networks not found")     
        else:
            return items
        
class GetNetwork(Resource):
    def get(self, netid):
        parser = reqparse.RequestParser()
        parser.add_argument('token')
        args = parser.parse_args()
        user = api_tools.challenge_token(args['token'])
        if netid in user.networkList:
            net = network.query.filter_by(id_network=netid).first()
            return net.to_dict()
        else:
            abort(403, message="Access denied")

class GetNetworkByName(Resource):
    def get(self, netname):
        parser = reqparse.RequestParser()
        parser.add_argument('token')
        args = parser.parse_args()
        user = api_tools.challenge_token(args['token'])
        net = network.query.filter_by(name=netname).first()
        if(net != None):
            if net.id_network in user.networkList:
                item = network.query.filter_by(id_network=net.id_network).first()
                return item.to_dict()
            else:
                abort(403, message="Access denied")
        else:
            abort(404, message="Network not found.")