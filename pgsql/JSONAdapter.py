#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
from pgsql.PostGreAdapter import PostGreAdapter
from io import StringIO
import pandas as pd
import datetime as dtime

def CustomFileRequest(dicoDevice, log,  param=None):
       
    headers = {"Content-Type": "application/json"}
    listDevice = list(dicoDevice.keys())
    print(listDevice)
    first=True
    for row in listDevice:
        print(row,dicoDevice[row])
        var = {
                  "authentication": {
                    "password": "hobolink",
                    "token": "ebf2c17beabc692f6bf4192ed64c3f32f0358f68",
                    "user": "marega77"
                  },
                  "query": "OPER_"+dicoDevice[row].upper()                
             }
        custom_date_parser = lambda x: dtime.datetime.strptime(x, "%y/%m/%d %H:%M:%S")
        params = json.dumps(var).encode("utf8")
        r = requests.post("https://webservice.hobolink.com/restv2/data/custom/file", data=params,headers=headers)
        print(dicoDevice[row],r.status_code)
        if(r.status_code == 200):
            df = pd.read_csv(StringIO(r.text), sep="\t",parse_dates=[0],date_parser=custom_date_parser)
            df.set_index(['Date'], inplace=True)
            if param != None:
                colList=[]
                for col in df.columns:
                    if param in col:
                         colList.append(col)
                df=df[colList]
            if first:
                dfTotal=df
                first=False
            else:
                dfTotal=pd.concat([dfTotal,df], axis=1)
             
        else:
            log.JSONFaultBadRequest(var)
        
        
    return (dfTotal) 

def JSONRequest(startDate, dicoDevice, log, hour, endDate):
    

    n = int((len(dicoDevice)/10)+2)
    listDevice = list(dicoDevice.keys())
    flist = []
    a = 0
    for i in range(1,n):
        sublist = listDevice[a:i*10]
        a = a+10
        
        flist.append(sublist)
        
    print(flist)
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
            print(x)
            pg = PostGreAdapter(log)
            pg.parseJSONData(x, hour, dicoDevice)
	    
        else:
            log.JSONFaultBadRequest(var)
        return (pg.temp) 