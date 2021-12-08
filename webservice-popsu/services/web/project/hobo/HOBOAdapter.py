#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .utils.DateFinder import findDate
from .pgsql.JSONAdapter import JSONRequest
from .utils.DeviceConfig import GetDicoDevice
from .utils.LogSystem import LogSystem
from datetime import datetime

class HoboAdapter:
    
    def __init__(self):
        self.log = LogSystem()
    
    
    def ImportData(self, hour, end_date):
        now = datetime.now()
        if(end_date == None):
            endDate = now.strftime("%Y-%m-%d %H:%M:%S")
        else:
            endDate = end_date

        dicoDevice = GetDicoDevice()
        date = findDate(endDate,hour, self.log)
        JSONRequest(date, dicoDevice, self.log, hour, endDate)