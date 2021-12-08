#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
from .PostGreAdapter import PostGreAdapter


def JSONRequest(startDate, dicoDevice, log, hour, endDate):
    

    n = int((len(dicoDevice)/10)+2)
    listDevice = list(dicoDevice.keys())
    flist = []
    a = 0
    for i in range(1,n):
        sublist = listDevice[a:i*10]
        a = a+10
        
        flist.append(sublist)
        
        
    for i in range(len(flist)):
    
        var = {
                  "action": "get_data",
                  "authentication": {
                    "password": "hobolink",
                    "token": "ebf2c17beabc692f6bf4192ed64c3f32f0358f68",
                    "user": "marega77"
                  },
                  "query": {
                    "end_date_time": endDate,
                    "loggers": flist[i],
                    "start_date_time": startDate
                  }
                }
      
    
        print(var)
        headers = {"Content-Type": "application/json"}
        params = json.dumps(var).encode("utf8")
        r = requests.post("https://webservice.hobolink.com/restv2/data/json", data=params,headers=headers)
        if(r.status_code == 200):
            sid = json.dumps(r.json())
            x = json.loads(sid)
            pg = PostGreAdapter(log)
            pg.parseJSONData(x, hour, dicoDevice)
        else:
            log.JSONFaultBadRequest(var)