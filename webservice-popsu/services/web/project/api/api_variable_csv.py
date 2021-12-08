#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask_restful import Resource, reqparse, abort
from ..models import network
from . import api_tools

class GetVariableByNetworkCSV(Resource):
    def get(self, netname):
        parser = reqparse.RequestParser()
        parser.add_argument('token')
        args = parser.parse_args()
        user = api_tools.challenge_token(args['token'])
        net = network.query.filter_by(name=netname).first()
        if(net != None):
            if net.id_network in user.networkList:
                query = "SELECT * FROM measures_variable WHERE id_network = {}".format(net.id_network)
                return api_tools.download_file(query)  
            else:
                abort(403, message="Access denied")   
        else:
            abort(404, message="Network not found")