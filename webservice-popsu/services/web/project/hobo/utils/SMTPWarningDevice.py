# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:31:01 2020

@author: Azelat
"""

import smtplib, ssl



class SMTPWarningDevice():
    
    def __init__(self):
        self.port = 465
        self.smtp_server = "smtp.gmail.com"
        self.sender_email = "POPSUProject@gmail.com"
        self.receiver_email = "corentin.faisy@iut-dijon.u-bourgogne.fr"  
        self.password = "hobopopsu"
        
    def JSONFaultWarn(self):
        message = """\
        Subject: Alerte HOBOAdapter
        
        Une erreur est survenue lors de l'envoie de la requête. Aucun résultat n'a été retourné."""

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.smtp_server, self.port, context=context) as server:
            server.login(self.sender_email, self.password)
            server.sendmail(self.sender_email, self.receiver_email, message.encode("utf-8"))
            
    def PGSQLFaultConnection(self):
        message = """\
        Subject: Alerte HOBOAdapter
        
        Une erreur est survenue lors de la connexion à la base de données. Connexion impossible."""

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.smtp_server, self.port, context=context) as server:
            server.login(self.sender_email, self.password)
            server.sendmail(self.sender_email, self.receiver_email, message.encode("utf-8"))            