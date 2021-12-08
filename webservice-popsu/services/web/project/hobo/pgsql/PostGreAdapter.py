#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import psycopg2
import pandas as pd
import os

class PostGreAdapter:
    
    def __init__(self, log):
        self.log = log
        self.connection = None
        try:
            self.connection = psycopg2.connect(dbname = "envirodb",user = os.environ["POSTGRES_USER"],host = os.environ["POSTGRES_HOST"],password = os.environ["POSTGRES_PASSWORD"])
            
            print("Connexion PostGre réussie")
            self.cursor = self.connection.cursor()

        except (Exception, psycopg2.Error) as error :
            print ("Error while connecting to PostgreSQL", error)
            self.log.PostGreBadConnection() 

    
    
    
    def parseJSONData(self, data, hour, dicoDevice):
        d = pd.DataFrame(data["observationList"])
        if(len(data["observationList"]) != 0):
        

            dfTemp = pd.DataFrame(d.loc[d["sensor_sn"].str.endswith('-1'), ["si_value", "logger_sn", "timestamp"]])  
            dfTemp.reset_index(inplace = True, drop=True)
            dfRH = pd.DataFrame(d.loc[d["sensor_sn"].str.endswith('-2'), ["si_value", "logger_sn", "timestamp"]])        
            dfRH.reset_index(inplace = True, drop=True)
            
            listDeviceTemp = []
            listDeviceRH = []
            
            for i in range(len(dfTemp)):
                listDeviceTemp.append(dicoDevice[int(dfTemp["logger_sn"][i])])
                
            for i in range(len(dfRH)):
                listDeviceRH.append(dicoDevice[int(dfRH["logger_sn"][i])])
                
            self.ExecuteQueryForTemp(dfTemp, listDeviceTemp)
            self.ExecuteQueryForHumidity(dfRH, listDeviceRH)


        else:
            self.log.JSONFaultNoData(data)
        
        
        print("Insertion effectuée avec succès.")
        self.cursor.close()
        self.connection.close()



        
    def ExecuteQueryForTemp(self, dfTemp, listDeviceTemp):
        
        
        for i in range(len(dfTemp)):
    
            request = """SELECT id_sensor FROM sensors WHERE name = %s"""
            self.cursor.execute(request, (listDeviceTemp[i],))
            
            id_sensor = self.cursor.fetchone()
            
            if(id_sensor != None):
                
                request = """SELECT * FROM measures WHERE id_station = %s AND date = %s AND measure_type = (SELECT id_type FROM measures_variable WHERE name = 'Température de l''air à 2m')"""
                self.cursor.execute(request, (id_sensor[0], dfTemp["timestamp"][i]))
                
                checkval = self.cursor.fetchone()
                if(checkval == None):
            
                    request = """SELECT id_network FROM sensors WHERE id_sensor = %s"""
                    self.cursor.execute(request, (id_sensor[0],))
                    
                    id_network = self.cursor.fetchone()
                    if(id_network != None):
                    
                        request = """SELECT * FROM measures_variable WHERE name = 'Température de l''air à 2m' AND id_network = %s"""
                        self.cursor.execute(request, (id_network[0],))
                        
                        checkval2 = self.cursor.fetchone()
                        
                        if(checkval2 == None):
                        
                            request = """INSERT into measures_variable (unite, name, id_network) VALUES ('°c', 'Température de l''air à 2m', %s)"""
                            self.cursor.execute(request, (id_network[0],))
                        
                        
                        request = """SELECT id_type from measures_variable WHERE name = 'Température de l''air à 2m' AND id_network = %s"""
                        self.cursor.execute(request, (id_network[0],))
                    
                        id_type = self.cursor.fetchone()
                        request = """INSERT INTO measures (measure, measure_type, id_station, date) VALUES (%s, %s, %s, %s)"""
                        self.cursor.execute(request, (dfTemp["si_value"][i], id_type[0], id_sensor[0], dfTemp["timestamp"][i]))
                    else:
                        self.log.PostgreNetworkNotFound(listDeviceTemp[i])
                    
                    
                else:
                    request = """UPDATE measures SET measure = %s WHERE id_station = %s AND date = %s"""
                    self.cursor.execute(request, (dfTemp["si_value"][i], id_sensor[0], dfTemp["timestamp"][i]))
                    
            else:
                self.log.PostgreStationUnknown(listDeviceTemp[i])
                    
        self.connection.commit()    
        
    def ExecuteQueryForHumidity(self, dfRH, listDeviceRH):
        
        
        for i in range(len(dfRH)):
    
            request = """SELECT id_sensor FROM sensors WHERE name = %s"""
            self.cursor.execute(request, (listDeviceRH[i],))
            
            id_sensor = self.cursor.fetchone()
            
            if(id_sensor != None):
                
                request = """SELECT * FROM measures WHERE id_station = %s AND date = %s AND measure_type = (SELECT id_type FROM measures_variable WHERE name = 'Humidité relative')"""
                self.cursor.execute(request, (id_sensor[0], dfRH["timestamp"][i]))
                
                checkval = self.cursor.fetchone()
                if(checkval == None):
            
                    
                    request = """SELECT id_network FROM sensors WHERE id_sensor = %s"""
                    self.cursor.execute(request, (id_sensor[0],))
                    
                    id_network = self.cursor.fetchone()
                    if(id_network != None):
                    
                        request = """SELECT * FROM measures_variable WHERE name = 'Humidité relative' AND id_network = %s"""
                        self.cursor.execute(request, (id_network[0],))
                        
                        checkval2 = self.cursor.fetchone()
                        
                        if(checkval2 == None):
                        
                            request = """INSERT into measures_variable (unite, name, id_network) VALUES ('%', 'Humidité relative', %s)"""
                            self.cursor.execute(request, (id_network[0],))
                        
                        
                        request = """SELECT id_type from measures_variable WHERE name = 'Humidité relative' AND id_network = %s"""
                        self.cursor.execute(request, (id_network[0],))
                        id_type = self.cursor.fetchone()                  
                    
                        request = """INSERT INTO measures (measure, measure_type, id_station, date) VALUES (%s, %s, %s, %s)"""
                        self.cursor.execute(request, (dfRH["si_value"][i], id_type[0], id_sensor[0], dfRH["timestamp"][i]))
                    else:
                        self.log.PostgreNetworkNotFound(listDeviceRH[i])
                    
                    
                else:
                    request = """UPDATE measures SET measure = %s WHERE id_station = %s AND date = %s"""
                    self.cursor.execute(request, (dfRH["si_value"][i], id_sensor[0], dfRH["timestamp"][i]))
                    
            else:
                self.log.PostgreStationUnknown(listDeviceRH[i])          

        self.connection.commit()   
        
        
        
