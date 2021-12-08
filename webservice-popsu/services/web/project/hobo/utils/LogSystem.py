#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:58:19 2020

@author: smartenv
"""

import logging
#from utils.SMTPWarningDevice import SMTPWarningDevice
from logging.handlers import RotatingFileHandler

  
class LogSystem:
    
    def __init__(self):
        self.logger = logging.getLogger()
        self.formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
        self.file_handler = RotatingFileHandler('project/hobo/log/SystemLog.log', 'a', 10000000, 1)
        #self.SMTP = SMTPWarningDevice()
    
    def JSONFaultBadRequest(self, requete):
        self.file_handler.setLevel(logging.ERROR)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.error("Requête incorrecte.\n\tRequête: %s\n",requete)
        #self.SMTP.JSONFaultWarn()
        
    def JSONWarningNoData(self, requete):
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.warning("Aucune donnée n'a été été récupérée.\n\tRequête: %s\n",requete)
    
    def WrongWindow(self, hour):
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.warning("Fenêtre temporelle incorrecte.\n\thour: %d\n",hour)
        
        
    def PostGreBadConnection(self):
        self.file_handler.setLevel(logging.ERROR)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.error("Connexion PosteGreSQL échouée.")
        #self.SMTP.PGSQLFaultConnection()     
        
    def CheckerLessData(self, nbFind, nbPredict, datatype):
        self.file_handler.setLevel(logging.WARNING)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.warning("Checker: Moins de données trouvées qu'espérées.\nType:%s\nTrouvées: %d\nEspérées:%d\n", datatype, nbFind, nbPredict)   
        
    def CheckerMoreData(self, nbFind, nbPredict, datatype):
        self.file_handler.setLevel(logging.WARNING)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.warning("Checker: Plus de données trouvées qu'espérées. Possible duplication de données\nTrouvées: %d\nEspérées:%d\n", datatype, nbFind, nbPredict)
        
    def DataDuplicate(self, nbData, device, date):
        self.file_handler.setLevel(logging.WARNING)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.warning("Checker: Duplication de données.\nTrouvées: %d\Capteur:%d\nDate:%d\n", nbData, device, date)  

    def PostgreStationUnknown(self, device):
        self.file_handler.setLevel(logging.WARNING)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.warning("Insertion: Aucune station n'est associée à la station %s ", device)    
        
        
    def PostgreNetworkNotFound(self, station):
        self.file_handler.setLevel(logging.ERROR)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.warning("Insertion: Aucun network associé à la station %s", station)           