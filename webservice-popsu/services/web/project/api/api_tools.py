#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask_restful import abort
from ..models import users
from flask import send_file
import psycopg2
import os

def abort_if_user_doesnt_exist(user_id):
    if(user_id == None):
        abort(404, message="An error is occured")    
    else:
        if(user_id < 0):
            if(user_id == -1):
                abort(404, message="Expired Token")
            if(user_id == -2):
                abort(404, message="Invalid Token")
            else:
                abort(404, message="An error is occured")


def challenge_token(token):

    auth_id = users.decode_auth_token(token)
    abort_if_user_doesnt_exist(auth_id)
    user = users.query.filter_by(id=auth_id).first()
    return user

def download_file(query):

    if os.path.exists("download/extract.csv"):
        os.remove("download/extract.csv")
    
    print("Generate CSV file...")
    
    connection= psycopg2.connect(dbname = "envirodb",user = os.environ["POSTGRES_USER"],host = os.environ["POSTGRES_HOST"],password = os.environ["POSTGRES_PASSWORD"])
    
    cursor = connection.cursor()

    SQL_for_file_output = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)

    t_path_n_file = "download/extract.csv"
    
    try:
        with open(t_path_n_file, 'a') as f_output:
            cursor.copy_expert(SQL_for_file_output, f_output)
    except psycopg2.Error as e:
        t_message = "Error: Bad extraction "+ str(e)
        return print(t_message)
    

    cursor.close()
    connection.close()    
    
    print("Download CSV file...")

    
    return send_file(os.path.abspath("download/extract.csv"), as_attachment=True)
