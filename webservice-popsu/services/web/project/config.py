import os


basedir = os.path.abspath(os.path.dirname(__file__))
user = os.environ['POSTGRES_USER']
password = os.environ['POSTGRES_PASSWORD']
host = os.environ['POSTGRES_HOST']
database = os.environ['POSTGRES_DB']
port = os.environ['POSTGRES_PORT']
secret_key = os.environ['SECRET_KEY']

DATABASE_CONNECTION_URI = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/envirodb'
DATABASE_USER_CONNECTION_URI = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'

STATIC_FOLDER = f"{os.getenv('APP_FOLDER')}/project/static"
MEDIA_FOLDER = f"{os.getenv('APP_FOLDER')}/project/media"

