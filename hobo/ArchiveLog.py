#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:17:26 2020

@author: smartenv
"""

import os
import datetime
import tarfile
import shutil
import os.path
from os import path

class ArchiveLog:
    
    def __init__(self):
        pass
    
    def startArchive(self):
        Current_Date = datetime.datetime.today().strftime ('%d-%b-%Y')        
        if path.exists("project/hobo/log/SystemLog.log"):
            os.rename(r'project/hobo/log/SystemLog.log',r'project/hobo/log/Log_' + str(Current_Date) + '.log')


        count = len(os.listdir('project/hobo/log'))
        if(count > 30):
            tz = tarfile.open('project/hobo/log/Logs_'+str(Current_Date)+'.tar.gz', 'w:gz')
            for file in os.listdir('project/hobo/log'):
                if(file.endswith(".txt") or file.endswith(".log")):
                    tz.add("project/hobo/log/"+file)
                    os.remove("project/hobo/log/"+file)
            tz.close()
            shutil.move('project/hobo/log/Logs_'+str(Current_Date)+'.tar.gz', 'project/hobo/log/Archives')