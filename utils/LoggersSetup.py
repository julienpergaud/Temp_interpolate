#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:15:40 2020

@author: smartenv
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--add", help="Permet d'ajouter un appareil à la liste, en passant son logger")
parser.add_argument("--remove", help="Permet de supprimer un appareil de la liste, en passant son logger")

args = parser.parse_args()

if (args.add != None):
    exist = 0
    file = open("project/hobo/loggers.txt", "r")
    loggers = file.read().split("\n")
    for lId in loggers:
        if lId == args.add:
            print("Cet appareil existe déjà.")
            exist = 1
            break
    file.close()
    if(exist == 0):
        file = open("project/hobo/loggers.txt", "a")
        file.write(args.add+"\n")
    file.close()

elif (args.remove != None):
    
    file = open("project/hobo/loggers.txt", "r")
    loggers = file.read().split("\n")
    listDevice = []
    for lId in loggers:
        if lId != args.remove:
            listDevice.append(lId)

    file.close()
    file = open("project/hobo/loggers.txt", "w")
    file.flush()
    for device in listDevice:
        file.write(device)    
    file.close()
    
file = open("project/hobo/loggers.txt", "r")
loggers = file.read().split("\n")
for line in loggers:
    print(line)
file.close()
            