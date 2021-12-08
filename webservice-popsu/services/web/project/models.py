#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from geoalchemy2.types import Geometry
#from geoalchemy2.elements import WKBElement
#from geoalchemy2.shape import to_shape
from sqlalchemy_serializer import SerializerMixin
import datetime
import jwt
from .config import secret_key


db = SQLAlchemy()


class measures_variable(db.Model, SerializerMixin):
    __bind_key__ = 'envirodb'   
    __tablename__ = 'measures_variable'
    id_type = db.Column(db.Integer(), primary_key=True)
    unite = db.Column(db.String(6))
    name = db.Column(db.String(32))
    id_network =  db.Column(db.Integer, nullable=True)


class jonction(db.Model, SerializerMixin):
    __bind_key__ = 'envirodb'
    __tablename__ = 'jonction'
    id_jonction = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(124))
    id_network = db.Column(db.Integer())
    sensor_name = db.Column(db.String(124))
    network_name = db.Column(db.String(124))
    var_id = db.Column(db.Integer())
    sensor_id = db.Column(db.Integer())


class measures(db.Model, SerializerMixin):
    __bind_key__ = 'envirodb'
    __tablename__ = 'measures'
    id_measure = db.Column(db.Integer(), primary_key=True)
    measure = db.Column(db.Float())
    measure_type = db.Column(db.Integer(), db.ForeignKey('measures_variable.id_type'), nullable=False)
    id_station = db.Column(db.Integer())
    date = db.Column(db.DateTime())

class network(db.Model, SerializerMixin):
    __bind_key__ = 'envirodb'
    __tablename__ = 'network'
    id_network = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(126))
    descCourt = db.Column(db.String(256), nullable=True)
    descLong = db.Column(db.String(1024), nullable=True)
    descCourtEn = db.Column(db.String(256), nullable=True)
    descLongEn = db.Column(db.String(1024), nullable=True)
    dateDebut = db.Column(db.DateTime(), nullable=True)
    dateFin = db.Column(db.DateTime(), nullable=True)


class sensors(db.Model, SerializerMixin):
    
    def serialize_int(value):
        return value + 100

#    serialize_types = (
#            (WKBElement, lambda x: to_shape(x).to_wkt()),
#            (int, serialize_int)
#        )

    __bind_key__ = 'envirodb'
    __tablename__ = 'sensors'
    
    
    id_sensor = db.Column(db.Integer(), primary_key=True)
    latitude = db.Column(db.Float(), nullable = True)
    longitude = db.Column(db.Float(), nullable = True)
    altitude = db.Column(db.Float(), nullable = True)
    name = db.Column(db.String(128), nullable = True)
    geometry = db.Column(Geometry(geometry_type='POINT', srid=4326))
    id_network = db.Column(db.Integer(), db.ForeignKey('network.id_network'), nullable = False)
    
    serialize_only = ('id_sensor', 'latitude', 'longitude', 'name', 'id_network')
    
class sensors_param(db.Model, SerializerMixin):
    __bind_key__ = 'envirodb'
    __tablename__ = 'sensors_param'
    id = db.Column(db.Integer(), primary_key = True)
    name = db.Column(db.String(124), nullable = True)
    value = db.Column(db.String(124), nullable = True)
    id_sensor = db.Column(db.Integer(), db.ForeignKey('sensors.id_sensor'))

class users(db.Model, SerializerMixin, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(42))
    pswd = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(128), unique=True, nullable=False)
    active = db.Column(db.Boolean(), default=True, nullable=False)
    role = db.Column(db.Integer(), nullable=True)
    admin = db.Column(db.Boolean(), nullable=False, default=False)
    networkList = db.Column(db.ARRAY(db.Integer()))
    token = db.Column(db.String(255) , nullable=True)
     
    serialize_only = ('id', 'name', 'email')
    
    def encode_auth_token(self, user_id):
        try:
            payload = {
                'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30, seconds=0),
                'iat': datetime.datetime.utcnow(),
                'sub': user_id
            }
            return jwt.encode(
                payload,
                secret_key,
                algorithm='HS256'
            )
        except Exception as e:
            return e

    def decode_auth_token(auth_token):
        try:
            payload = jwt.decode(auth_token, secret_key)
            return payload['sub']
        except jwt.ExpiredSignatureError:
            return -1
        except jwt.InvalidTokenError:
            return -2