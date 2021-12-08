#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:57:27 2020

@author: smartenv
"""

import pandas as pd

def SetDeviceList(argdevice):
    listDevice2 = []
    if argdevice != None:
        listDevice = argdevice.split(",")
        for device in listDevice:
            listDevice2.append(int(device))
    else:
        with open("project/hobo/loggers.txt", "r") as file:
            for line in file.readlines():
                listDevice2.append(int(line))
    
    return listDevice2


def GetDicoDevice():
    data = pd.read_csv("project/hobo/utils/HOBOLoggers.csv",sep=',')
    df = pd.DataFrame(data)
    
    df2 = pd.DataFrame(df.loc[df["logger"] != 0])
    df2.reset_index(drop=True, inplace=True)
    
    
    dicoDevice = {}
    
    listLogger = df2["logger"].to_list()
    
    listDevice = df2["sensor_id"].to_list()
    
    
    for i in range(len(listLogger)):
        dicoDevice[listLogger[i]] = listDevice[i]
    
    return dicoDevice