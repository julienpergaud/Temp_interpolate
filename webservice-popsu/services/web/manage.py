# coding : utf8

from flask.cli import FlaskGroup
from passlib.hash import sha256_crypt
from project import app
import click
from psycopg2 import connect, extensions
from project.models import db, users, network, measures_variable
from project.DataImporter import DataImporter
from project.hobo import HOBOAdapter
from project.hobo import ArchiveLog
import os as os
cli = FlaskGroup(app)


@cli.command("init_db", help="Inits one database for example")
def init_db():
    con = connect(dbname = "dbuser",user = os.environ["POSTGRES_USER"],host = os.environ["POSTGRES_HOST"],
            password = os.environ["POSTGRES_PASSWORD"])
    cursor = con.cursor()
    autocommit = extensions.ISOLATION_LEVEL_AUTOCOMMIT
    con.set_isolation_level( autocommit )
    sqlCreateDatabase = "create database envirodb;"
    cursor.execute(sqlCreateDatabase)
    cursor.close()
    con.close()
   
    init_postgis("envirodb")
    
    db.create_all(bind='envirodb')
    db.session.commit()
    
    
@cli.command("init_db2", help="Inits all database, and adds some example values")
def init_db_2():
    con = connect(dbname = "dbuser",user = os.environ["POSTGRES_USER"],host = os.environ["POSTGRES_HOST"],password = os.environ["POSTGRES_PASSWORD"])
    cursor = con.cursor()
    autocommit = extensions.ISOLATION_LEVEL_AUTOCOMMIT
    con.set_isolation_level( autocommit )
    sqlCreateDatabase = "create database envirodb;"
    cursor.execute(sqlCreateDatabase)
    cursor.close()
    con.close()
    
    init_postgis("envirodb")
    
    db.create_all(bind='envirodb')
    db.session.commit()
    
    create_network("network_1")
    add_variable_type("°c", "temperature", "1")
    add_device_from_csv("project/ressources/METADATA_MUSTARD_2019.csv",1,',')
    add_data_from_csv("project/ressources/enviroHRexample.csv",';',1)

    
@cli.command("create_admin", help="Creates an administrator user")
def create_admin():
    admin = users(name="admin", pswd=sha256_crypt.encrypt("123456"),email="envirodb.ufr@gmail.com", active=True, admin=True)
    db.session.add(admin)
    db.session.commit()
    token = admin.encode_auth_token(admin.id)
    admin.token = token.decode("utf-8")
    db.session.commit()
   
@cli.command("create_user", help="Creates an user administrator with specific name and password")
@click.argument("name")
@click.argument("pswd")
def create_user(name, pswd):
    
    user = users(name=name, pswd=sha256_crypt.encrypt(pswd),email=name,active=True, admin=True)
    db.session.add(user)
    db.session.commit()
    token = user.encode_auth_token(user.id)
    user.token = token.decode("utf-8")
    db.session.commit()
    
@cli.command("create_network", help="Creates network with name as parameter")
@click.argument("name")
def cmd_network(name):
    create_network(name)
    
@cli.command("drop_tables", help="Drops all tables from the database")   
def drop_tables_all():
    db.drop_all()
    db.session.commit()

@cli.command("drop_db", help="Drops envirodb database")
def drop_db_all():
    con = connect(dbname = "dbuser",user = os.environ["POSTGRES_USER"],host = os.environ["POSTGRES_HOST"],password = os.environ["POSTGRES_PASSWORD"])
    
    cursor = con.cursor()   
    autocommit = extensions.ISOLATION_LEVEL_AUTOCOMMIT
    con.set_isolation_level( autocommit )
    sqlCreateDatabase = "Drop database envirodb;"
    cursor.execute(sqlCreateDatabase)
    cursor.close()
    con.close()   
    
@cli.command("add_variable_ex", help="Adds unite parameter for measures, with unite, name, and network as arguments")
@click.argument("unite")
@click.argument("name")
@click.argument("id_network")
def cmd_add_variable(unite, name, id_network):
    add_variable_type(unite, name, id_network)
    
@cli.command("parse_csv_data", help ="Reads & imports measurement data from csv to the database")
@click.argument("name")
@click.argument("network")
@click.argument("separator")
def cmd_csv_data(name, separator, network):
    add_data_from_csv(name, separator, network)
    
    
@cli.command("parse_csv_device", help ="Reads & imports sensors data from csv to the database, with file's path, network to associate, separator and sensors parameters as arguments, csv file must contain nom, lat and long column as prerequites")
@click.argument("name")
@click.argument("network")
@click.argument("separator")
def cmd_csv_device(name, network, separator):
    add_device_from_csv(name, network, separator)
    
    
@cli.command("init_postgis", help="Init postGIS to the specified database")
@click.argument("dbname")
def cmd_postgis(dbname):
    init_postgis(dbname)
    
    
@cli.command("extract_csv", help="Extracts data from the database to csv file")
@click.argument("table_name")
def extract_csv(table_name):
    extract_data(table_name)

@cli.command("update_jonction", help="Update jonction table with data from sensors, measures and variables")
@click.argument("id_network")
def cmd_update_jonction(id_network):
    d = DataImporter()
    d.UpdateJonction(id_network)
    d.close_connection()

@cli.command("hobo", help="Force the update for all hobo's data")
@click.argument("time_window")
@click.argument("end_date", nargs=-1, type=click.Path())
def cmd_execute_hobo_update(time_window, end_date):
    hobo = HOBOAdapter.HoboAdapter()
    hobo.ImportData(int(time_window), end_date[0])

@cli.command("archivelog", help="Forces the archive of all log of HOBOIntegration")    
def cmd_archive():
    archive = ArchiveLog()
    archive.startArchive()

@cli.command("drop_data", help="Drop all data and variables from database")
def drop_data():
    dropData()

@cli.command("close_session", help="Closes all sessions on Database")
def close_session():
    con = connect(dbname = "dbuser",user = os.environ["POSTGRES_USER"],host = os.environ["POSTGRES_HOST"],password = os.environ["POSTGRES_PASSWORD"])
    cursor = con.cursor()
    autocommit = extensions.ISOLATION_LEVEL_AUTOCOMMIT
    con.set_isolation_level( autocommit )

    request = """SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = 'envirodb';"""

    cursor.execute(request)
    cursor.close()
    print("Sessions closed")
    con.close()


def dropData():
    con = connect(dbname = "envirodb",user = os.environ["POSTGRES_USER"],host = os.environ["POSTGRES_HOST"],password = os.environ["POSTGRES_PASSWORD"])
    cursor = con.cursor()
    autocommit = extensions.ISOLATION_LEVEL_AUTOCOMMIT
    con.set_isolation_level( autocommit )
    sql = "Delete FROM measures;"
    cursor.execute(sql)

    sql = "Delete FROM measures_variable;"
    cursor.execute(sql)
    cursor.close()
    print("Data deleted")
    con.close()



def add_variable_type(unite, name, id_network):
    variable = measures_variable(unite=unite, name=name, id_network=id_network)
    db.session.add(variable)
    db.session.commit()

    
def add_data_from_csv(name, separator, network):
    d = DataImporter()
    d.parseCSVForData(name, separator, network)
    d.close_connection()
    
    
    
def add_device_from_csv(nom,network,separator):
    d = DataImporter()
    d.parseCSVForDevice(nom,network,separator)
    d.close_connection()
    
    
def create_network(name):
    reseau = network(name=name, descCourt="Réseau test", descLong="Aucun")
    db.session.add(reseau)
    db.session.commit()
    
    
def init_postgis(dbname):
    con = connect(dbname = dbname,user = os.environ["POSTGRES_USER"],host = os.environ["POSTGRES_HOST"],password = os.environ["POSTGRES_PASSWORD"])
    cursor = con.cursor()   
    autocommit = extensions.ISOLATION_LEVEL_AUTOCOMMIT
    con.set_isolation_level( autocommit )
    sqlCreateDatabase = "CREATE EXTENSION postgis;"
    cursor.execute(sqlCreateDatabase)
    cursor.close()
    con.close()     

def extract_data(table):
    d = DataImporter()
    d.ExtractData(table)
    d.close_connection()

if __name__ == "__main__":
    cli()
