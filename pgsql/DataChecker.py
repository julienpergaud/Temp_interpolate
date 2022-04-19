#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:45:35 2020

@author: smartenv
"""



class DataChecker:
    
    def __init__(self, connection, logger):
        self.connection = connection
        self.cursor = self.connection.cursor()
        self.logger = logger
        self.delta = 3
        
        
    def CheckTemp(self, dfTemp, nbHour):
        
        device = []
        for i in range(len(dfTemp)):
            device.append(dfTemp["logger_sn"][i])

        device = list(set(device))
        print(device)
        
        
        for i in range(len(device)):
            request = """SELECT count(*) FROM ct.temp WHERE id_station = {}""" 
            self.cursor.execute(request.format(device[i]))
            cpt = self.cursor.fetchone()
            if(cpt[0] < nbHour-self.delta):
                print("Données manquantes sur: ",device[i])
                self.logger.CheckerLessData(cpt[0], nbHour, "Température")
            elif(cpt[0] > nbHour+self.delta):
                print("Données en surplus sur: ",device[i])
                self.logger.CheckerMoreData(cpt[0], nbHour, "Température")
            else:
                pass
                    
            
    
    def CheckRH(self, dfRH, nbHour):
        device = []
        for i in range(len(dfRH)):
            device.append(dfRH["logger_sn"][i])
            
        device = list(set(device))
        print(device)
        
        
        for i in range(len(device)):
            request = """SELECT count(*) FROM ct.temp WHERE id_station = {}""" 
            self.cursor.execute(request.format(device[i]))
            cpt = self.cursor.fetchone()
            if(cpt[0] < nbHour-self.delta):
                print("Donnée manquante sur: ",device[i])
                self.logger.CheckerLessData(cpt[0], nbHour, "RH")
            elif(cpt[0] > nbHour+self.delta):
                print("Donnée en surplus sur: ",device[i])
                self.logger.CheckerMoreData(cpt[0], nbHour, "RH")
            else:
                pass
            
    def CheckDuplication(self, dfTemp):
        
        device = []
        date = []
        for i in range(len(dfTemp)):
            device.append(dfTemp["logger_sn"][i])
            date.append(dfTemp["timestamp"][i])
            
        for i in range(len(device)):
            for j in range(len(date)):
                request = """SELECT count(*) FROM ct.temp WHERE id_station = {} AND date = '{}'"""
                self.cursor.execute(request.format(device[i], date[i]))
                cpt = self.cursor.fetchone()
                if(cpt[0] > 1):
                    print("Dupplication de données: ", device[i], date[i])
                    self.logger.DataDuplicate(cpt[0], device[i], date[i])