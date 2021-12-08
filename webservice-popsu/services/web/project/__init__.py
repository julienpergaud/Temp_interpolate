from flask import Flask
from . import config

def create_app():
    flask_app = Flask(__name__)
    flask_app.config['SECRET_KEY'] = 'jesuisunecleftressecrete'
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = config.DATABASE_USER_CONNECTION_URI
    flask_app.config['SQLALCHEMY_BINDS'] = {'envirodb': config.DATABASE_CONNECTION_URI}
    flask_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    flask_app.app_context().push()

    return flask_app
