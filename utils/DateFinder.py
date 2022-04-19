#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from datetime import timedelta

def findDate(enddate,argHour, log):
    if(argHour > 0):
        startdate=datetime.strptime(enddate,"%Y-%m-%d %H:%M:%S")
        startdate=startdate-timedelta(hours=argHour)
        startdate=startdate.strftime("%Y-%m-%d %H:%M:%S")
        return startdate
    else:
        log.WrongWindow(argHour)


    
def calculHour(argHour, now):
    if(now.hour >= argHour):
        return abs(now.hour - argHour)
    else:
        return abs((now.hour + argHour) % 24)
