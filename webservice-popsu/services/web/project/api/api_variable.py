#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask_restful import Resource, reqparse, abort
from ..models import network, measures_variable
from . import api_tools

class GetVariableByNetwork(Resource):
    def get(self, netname):
        parser = reqparse.RequestParser()
        parser.add_argument('token')
        args = parser.parse_args()
        user = api_tools.challenge_token(args['token'])
        net = network.query.filter_by(name=netname).first()
        if(net != None):
                
            items = []
            if net.id_network in user.networkList:
                variables = measures_variable.query.filter_by(id_network=net.id_network).all()
                for item in variables:
                    items.append(item.to_dict())
                
                return items
            else:
                abort(403, message="Access denied")   
        else:
            abort(404, message="Network not found")