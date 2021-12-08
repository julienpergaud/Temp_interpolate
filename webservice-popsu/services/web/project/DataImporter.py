#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import psycopg2
import os as os
import time
import psycopg2.extras as psyex

class DataImporter:
    
    def __init__(self):
        self.connection = None
        try:
            self.connection=psycopg2.connect(dbname="envirodb", user = os.environ["POSTGRES_USER"], password = os.environ["POSTGRES_PASSWORD"], host=os.environ["POSTGRES_HOST"]) 
            # print("Connexion PostGre réussie")
            self.cursor = self.connection.cursor()

        except (Exception, psycopg2.Error) as error :
            print ("Error while connecting to PostgreSQL", error)

    def close_connection(self):
        self.cursor.close()
        self.connection.close()
    
    
    def parseCSVForData(self, file, separator, network):
        data = pd.read_csv(file,sep=separator)
        df = pd.DataFrame(data)
    
        self.ExecuteQueryForData2(df, network)
        self.connection.close()

    def parseCSVForDevice(self, file, network, separator):
        data = pd.read_csv(file,sep=separator)
        df = pd.DataFrame(data)
        self.ExecuteQueryForDevice(df, network)
        
        
    def ExecuteQueryForDevice(self, df, network):
        
        request = """SELECT id_network FROM network WHERE id_network = {} """
        query = request.format(network)
        self.cursor.execute(query)
        n = self.cursor.fetchone()
        
        if(n != None):
            res = n[0]
        else:
            print("erreur: Réseau introuvable.")
               
        parameter=df.columns

        for i in range(len(df)):
            request = "SELECT * FROM sensors WHERE name = %s AND id_network = %s"
            
            self.cursor.execute(request, (df["NOM"][i], res))
            sensor = self.cursor.fetchone()
            
            if(sensor != None):
                #print("SQL check: Sensor already exists... Updating...")
                request = """UPDATE sensors SET latitude = %s, longitude = %s, geometry = ST_SetSRID(ST_POINT(%s, %s), 4326) WHERE name = %s AND id_network = %s"""
                self.cursor.execute(request, (df["LAT"][i], df["LON"][i], df["LON"][i], df["LAT"][i], df["NOM"][i], res)) 
                
                request = """SELECT id_sensor FROM sensors WHERE name = %s AND id_network = %s"""
                self.cursor.execute(request, (df["NOM"][i], res))
                n2 = self.cursor.fetchone()
                
                for param in parameter:
                    if param !='NOM' and param !='LAT' and param!='LON':
                        request = """SELECT * FROM sensors_param WHERE name = %s AND id_sensor = %s"""
                        self.cursor.execute(request, (param, n2[0]))
                        p = self.cursor.fetchone()
                        
                        if(p != None):
                            #print("SQL check: Sensor_param already exists... Updating...")  
                            request = """UPDATE sensors_param SET value = %s WHERE name = %s AND id_sensor = %s"""
                            self.cursor.execute(request, (str(df[param][i]), param, n2[0]))
                            
                        else:
                            request = """INSERT INTO sensors_param (name, value, id_sensor) VALUES (%s, %s, %s)"""
                            self.cursor.execute(request, (param, str(df[param][i]), n2[0]))
    
    
            else:
                request = """INSERT INTO sensors (latitude, longitude, name, geometry, id_network) VALUES (%s, %s, %s, ST_SetSRID(ST_POINT(%s, %s), 4326), %s)"""
                self.cursor.execute(request, (df["LAT"][i], df["LON"][i], df["NOM"][i], df["LON"][i], df["LAT"][i], res))
                    
                request = """SELECT id_sensor FROM sensors WHERE name = '{}' AND id_network = {}"""
                query = request.format(df["NOM"][i], res)
                self.cursor.execute(query)
                n2 = self.cursor.fetchone()
                
                for param in parameter:
                    if param !='NOM' and param !='LAT' and param!='LON':
                        request = """SELECT * FROM sensors_param WHERE name = %s AND id_sensor = %s"""
                        self.cursor.execute(request, (param, n2[0]))
                        p = self.cursor.fetchone()
                        
                        if(p != None):
                            #print("SQL check: Sensor_param already exists... Updating...")  
                            request = """UPDATE sensors_param SET value = %s WHERE name = %s AND id_sensor = %s"""
                            self.cursor.execute(request, (str(df[param][i]), param, n2[0]))
                            
                        else:
                            request = """INSERT INTO sensors_param (name, value, id_sensor) VALUES (%s, %s, %s)"""
                            self.cursor.execute(request, (param, str(df[param][i]), n2[0]))
            
        
        
        self.connection.commit()
        
        
    def ExecuteQueryForData2(self, df, network):
        
        listSensor = list(set(df["logger_name"]))
        listVariable = list(set(df["varunit"]))
        listVariable2 = list(set(df["varname"]))
        
        dicoSensor = {}
        dicoVariable = {}
        

        for i in range(len(listVariable)):
            request = """SELECT id_type FROM measures_variable WHERE unite LIKE %s AND name LIKE %s"""
            self.cursor.execute(request, (listVariable[i], listVariable2[i]))     
            typemesure = self.cursor.fetchone()
            
            if(typemesure == None):
                request = """INSERT INTO measures_variable (unite, name, id_network) VALUES (%s, %s, %s)"""
                self.cursor.execute(request, (listVariable[i], listVariable2[i], network))
                
                request = """SELECT id_type FROM measures_variable WHERE unite LIKE %s AND name LIKE %s"""
                self.cursor.execute(request, (listVariable[i], listVariable2[i]))     
                typemesure = self.cursor.fetchone()
                
                dicoVariable[listVariable2[i]]=typemesure[0]
                
            else:
                dicoVariable[listVariable2[i]]=typemesure[0]
                
                
        for i in range(len(listSensor)):
            request = """SELECT id_sensor FROM sensors WHERE name LIKE %s AND id_network = %s"""
            self.cursor.execute(request, (listSensor[i], network))
            id_station = self.cursor.fetchone()
            dicoSensor[listSensor[i]]=id_station[0]
         
        def func(x,y): 
            return y[x]
        df['id_station']=df['logger_name'].apply(func, y=dicoSensor)
        df['measure_type']=df['varname'].apply(func, y=dicoVariable)
        print(df)
        sql="CREATE UNLOGGED TABLE temp_data (date timestamp, name VARCHAR, value NUMERIC, namevar VARCHAR, nameunit VARCHAR, id_station INTEGER, measure_type INTEGER);"
        self.cursor.execute(sql)
         


        tpls = [tuple(x) for x in df.to_numpy()]

        sql = """INSERT INTO temp_data (date,name, value,namevar,nameunit,id_station,measure_type) VALUES (%s,%s,%s,%s,%s,%s,%s);"""    
        psyex.execute_batch(self.cursor,sql,tpls)



        sql="""INSERT INTO measures (measure,measure_type,id_station,date)
            sELECT  value, measure_type, id_station, date from temp_data;"""
        
        self.cursor.execute(sql)
        sql=""" DROP TABLE temp_data;"""
        self.cursor.execute(sql)

        self.connection.commit()


    def ExecuteQueryForData(self, df, network):
        
        listSensor = list(set(df["logger_name"]))
        listVariable = list(set(df["varunit"]))
        listVariable2 = list(set(df["varname"]))
        
        
        dicoSensor = {}
        dicoVariable = {}
        

        for i in range(len(listVariable)):
            request = """SELECT id_type FROM measures_variable WHERE unite LIKE %s AND name LIKE %s"""
            self.cursor.execute(request, (listVariable[i], listVariable2[i]))     
            typemesure = self.cursor.fetchone()
            
            if(typemesure == None):
                request = """INSERT INTO measures_variable (unite, name, id_network) VALUES (%s, %s, %s)"""
                self.cursor.execute(request, (listVariable[i], listVariable2[i], network))
                
                request = """SELECT id_type FROM measures_variable WHERE unite LIKE %s AND name LIKE %s"""
                self.cursor.execute(request, (listVariable[i], listVariable2[i]))     
                typemesure = self.cursor.fetchone()
                
                dicoVariable[listVariable2[i]] = typemesure[0]
                
            else:
                dicoVariable[listVariable2[i]] = typemesure[0]
                
                
        for i in range(len(listSensor)):
                request = """SELECT id_sensor FROM sensors WHERE name LIKE %s AND id_network = %s"""
                self.cursor.execute(request, (listSensor[i], network))
                id_station = self.cursor.fetchone()
                if(id_station == None):
                    df = pd.DataFrame(df.loc[df["logger_name"] != listSensor[i]])
                else:
                    dicoSensor[listSensor[i]] = id_station[0]
        
        df.reset_index(inplace=True)
                
        c = time.time()
        argument_string = ",".join("(%s, %s, %s, '%s')" % (df["si_value"][i], dicoVariable[df["varname"][i]], dicoSensor[df["logger_name"][i]], df["timestamp"][i]) for i in range(len(df)))
        self.cursor.execute("INSERT INTO measures (measure, measure_type, id_station, date) VALUES"+ argument_string)
        self.connection.commit()
        print("insertion terminée: ",time.time()-c)           
        
        
        self.connection.commit()   

    
    def CalculGeometry(self):
        query = """UPDATE sensors SET geometry = ST_SetSRID(ST_POINT(longitude, latitude), 4326)"""
        self.cursor.execute(query)
        self.connection.commit()
        
        
    def ExtractData(self, table):

        query = """SELECT * FROM "{}" """
        request = query.format(table)
    
        SQL_for_file_output = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(request)
    
        t_path_n_file = "project/ressources/" + table + ".csv"
    
        try:
            with open(t_path_n_file, 'w+') as f_output:
                self.cursor.copy_expert(SQL_for_file_output, f_output)
        except psycopg2.Error as e:
            s = "{}".format(e)
            t_message = "Error: " + s
            return print(t_message)

    def UpdateJonction(self, id_net):
        
        query = """DELETE FROM jonction WHERE id_network = %s"""
        self.cursor.execute(query, (id_net))       
        
        query = """INSERT INTO jonction(name, id_network, sensor_name, network_name, var_id, sensor_id)
                SELECT measures_variable.name,network.id_network,sensors.name,network.name,measures_variable.id_type,sensors.id_sensor FROM measures_variable 
                INNER JOIN network on network.id_network = measures_variable.id_network 
                INNER JOIN sensors on network.id_network = sensors.id_network and network.id_network=%s"""
                
        self.cursor.execute(query, (id_net))
        
        self.connection.commit()
