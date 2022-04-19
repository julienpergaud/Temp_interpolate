#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pgsql.JSONAdapter import CustomFileRequest
from utils.DeviceConfig import GetDicoDevice
from utils.LogSystem import LogSystem

class HoboAdapter:
    
    def __init__(self):
        self.log = LogSystem()
        self.data = None 
    
    def downloadtData(self):
        dicoDevice = GetDicoDevice()
        self.data=CustomFileRequest( dicoDevice, self.log, param="Temperature")
	