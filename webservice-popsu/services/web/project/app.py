from flask import render_template, request, redirect, flash, make_response, Response
from flask_cas import CAS, login, logout
from passlib.hash import sha256_crypt
from flask_login import LoginManager, login_user, login_required, current_user, logout_user
from flask_mail import Mail, Message
from flask_jsglue import JSGlue
from .models import db, users, network, sensors, measures_variable
from . import create_app
from datetime import datetime
from werkzeug.utils import secure_filename
from sqlalchemy import exc
from psycopg2 import connect
from .api.api_main import init_api
from .hobo.HOBOAdapter import HoboAdapter
from .hobo.ArchiveLog import ArchiveLog
from .DataImporter import DataImporter
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import re
import json
import os
import psycopg2
import logging
import  sys

UPLOAD_FOLDER = '/'
ALLOWED_EXTENSIONS = {'csv'}
logging.basicConfig(level=logging.DEBUG)

"""app = Flask(__name__)"""
app = create_app()
api = init_api(app)
#cron = BackgroundScheduler()




def hobo_update():
    print("Tâche automatique: Script Extraction HOBO...")
    hobo = HoboAdapter()
    hobo.ImportData(24, None)


def archive_log_hobo():
    archive = ArchiveLog()
    archive.startArchive()
    

def update_jonction():
    print("Tâche automatique: Update Jonction...")
    net = network.query.all()
    for n in net:
        DataImporter.UpdateJonction(n.id_network)
        
        

#cron.add_job(hobo_update, 'interval', hours=6)
#cron.add_job(archive_log_hobo, 'interval', hours=24)
#cron.add_job(update_jonction, 'interval', hours=12)
#cron.start()


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db.init_app(app)
db.create_all(bind=None)
cas = CAS()
cas.init_app(app, '/cas/')

login_manager = LoginManager()
login_manager.login_view = 'rqstlog'
login_manager.init_app(app)


mail = Mail(app)

jsglue = JSGlue()
jsglue.init_app(app)

#atexit.register(lambda: cron.shutdown(wait=False))

def jsonreader():
    configpath = os.path.abspath(os.path.dirname("config.json")) + "/config.json"
    basedir = os.path.abspath(os.path.dirname(__file__))
    data_file = os.path.join(basedir, 'static/js/config.json')
    configfile = {}
    with open(data_file) as json_file:
        data = json.load(json_file)
    return data

PARAMS = jsonreader()

app.config['MAIL_SERVER'] = PARAMS['config']['smtp']['MAIL_SERVER']
app.config['MAIL_PORT'] = int(PARAMS['config']['smtp']['MAIL_PORT'])
app.config['MAIL_USE_TLS'] = bool(PARAMS['config']['smtp']['MAIL_USE_TLS'])
app.config['MAIL_USE_SSL'] = bool(PARAMS['config']['smtp']['MAIL_USE_SSL'])
app.config['MAIL_DEBUG'] = bool(PARAMS['config']['smtp']['MAIL_DEBUG'])
app.config['MAIL_USERNAME'] = PARAMS['config']['smtp']['MAIL_USERNAME']
app.config['MAIL_PASSWORD'] = PARAMS['config']['smtp']['MAIL_PASSWORD']
app.config['MAIL_DEFAULT_SENDER'] = PARAMS['config']['smtp']['MAIL_SERVER']
app.config['MAIL_MAX_EMAILS'] = PARAMS['config']['smtp']['MAIL_SERVER']
app.config['MAIL_SUPPRESS_SEND'] = bool(PARAMS['config']['smtp']['MAIL_SERVER'])
app.config['MAIL_ASCII_ATTACHEMENTS'] = bool(PARAMS['config']['smtp']['MAIL_SERVER'])

MAX_GRAPH = int(PARAMS['config']['params']['nbgraph'])
CAS = PARAMS['config']['params']['cas']

@app.context_processor
def give_logo():
    configfile = jsonreader()
    return dict(hcontent=configfile['config']['header']['content'], link=configfile['config']['footer']['link'], linkMotif=configfile['config']['footer']['linkMotif'],
                flogo1=configfile['config']['footer']['files'][0], flogo2=configfile['config']['footer']['files'][1],flogo3=configfile['config']['footer']['files'][2],
                fcontent=configfile['config']['footer']['content'])


def getLang():
    if not request.cookies.get('LANG'):
        LANG = "FR"
    else:
        LANG = request.cookies.get('LANG')
    return LANG



@login_manager.user_loader
def load_user(user_id):
    return users.query.get(int(user_id))


@app.route('/french')
def french():
    resp = make_response(redirect('/'))
    resp.set_cookie('LANG', 'FR')
    return resp


@app.route('/english')
def english():
    resp = make_response(redirect('/'))
    resp.set_cookie('LANG', 'EN')
    return resp


@app.route('/',methods=['GET'])
def home():
    LANG = getLang()
    _reseaux = network.query.all()
    _capteurs = sensors.query.all()

    capteurs = {}

    for capt in _capteurs:
        capteurs[capt.id_sensor] = {}
        capteurs[capt.id_sensor]['id_network'] = capt.id_network
        capteurs[capt.id_sensor]['id_sensor'] = capt.id_sensor
        capteurs[capt.id_sensor]['latitude'] = capt.latitude
        capteurs[capt.id_sensor]['longitude'] = capt.longitude
        capteurs[capt.id_sensor]['name'] = capt.name

    if not current_user.is_authenticated:
        return render_template("pages/home" + LANG + ".html", log=False, reseaux=_reseaux, capteurs=capteurs)

    return render_template("pages/home" + LANG + ".html", log=True, user=current_user, admin=current_user.admin,
                           reseaux=_reseaux, capteurs=capteurs)


@app.route('/reqMailNetwork/<int:id_network>/<string:email>')
def reqMailNetwork(id_network, email):
    admin = users.query.filter_by(admin=True).first()
    msg = Message("Demande d'ajout à un réseau", recipients=[admin.email])
    msg.body = "Test email"
    mail.send(msg)
    return redirect('/')


@app.route('/login/<string:type>', methods=['GET', 'POST'])
def login(type):
    LANG = getLang()
    """if request.content_type is None:
        r = 'local'
    else:
        form_res = request.form
        r = form_res['CAS']"""

    return render_template('pages/login' + LANG + '.html', resultat=type, err='false')


@app.route('/rqstcas', methods=['POST'])
def rqstcas():
    LANG = getLang()
    form_res = request.form
    univ = form_res['univ']
    app.config['CAS_SERVER'] = univ
    app.config['CAS_AFTER_LOGIN'] = 'accountcas'

    return redirect('http://cas.u-bourgogne.fr')


@app.route('/rqstlog', methods=['POST'])
def rqstlog():
    LANG = getLang()
    form_res = request.form
    login = form_res['login']
    user = users.query.filter_by(name=login).first()
    if user is None:
        return render_template('pages/loginFR.html', resultat='local', err='1')

    mdp = form_res['mdp']
    if not sha256_crypt.verify(mdp, user.pswd):
        return render_template('pages/loginFR.html', resultat='local', err='1')

    if not user.active:
        return render_template('pages/loginFR.html', resultat='local', err='2')

    login_user(user, remember='remember')
    if user.admin is True:
        networksTab = []
        _networks = network.query.all()
        for net in _networks:
            networksTab.append(net.id_network)
        db.session.query(users).filter(users.id == user.id).update({users.networkList: networksTab}, synchronize_session=False)
        db.session.commit()

    return redirect('/')


@app.route('/register')
def register():
    LANG = getLang()
    return render_template('pages/register' + LANG + '.html', err='false', errlog='')


@app.route('/rqstreg', methods=['POST'])
def rqstreg():
    LANG = getLang()
    form_res = request.form
    _login = form_res['login']
    _email = form_res['email']
    _mdp1 = form_res['mdp1']
    _mdp2 = form_res['mdp2']
    _errlog = ""

    if len(_login) < 4 | len(_login) > 42:
        _errlog = "Votre pseudonyme est trop long ou trop court. -"

    checkuser1 = users.query.filter_by(name=_login).first()

    if not checkuser1 is None:
        _errlog += "Votre pseudonyme est deja utilisé. -"

    if not re.match('[^@]+@[^@]+\.[^@]+', _email):
        _errlog += "Votre email n'est pas valide -"

    checkuser2 = users.query.filter_by(email=_email).first()
    if not checkuser2 is None:
        _errlog += 'Votre email est deja utilisé -'

    if _mdp1 != _mdp2:
        _errlog += 'Les deux mots de passe ne sont pas identiques -'

    if len(_mdp1) < 4 | len(_mdp1) > 42:
        _errlog = "Votre mot de passe est trop long ou trop court. -"

    if _errlog != "":
        return render_template('pages/register' + LANG + '.html', err='true', errlog=_errlog)

    _mdp = sha256_crypt.encrypt(_mdp1)
    newuser = users(name=_login, pswd=_mdp, email=_email, active=False, admin=False)
    db.session.add(newuser)
    db.session.commit()

    admin = users.query.filter_by(admin=True).first()

    msg = Message("Demande d'inscription", recipients=[admin.email])
    msg.body = "L'utiliasteur " + _login + "souhaite s'inscrire sur EnviroDB"
    mail.send(msg)

    return redirect('/')


@app.route('/account')
@login_required
def account():
    LANG = getLang()
    return render_template('pages/account' + LANG + '.html', user=current_user)

@app.route('/modaccount',methods=['POST'])
@login_required
def modaccount():
    LANG = getLang()
    form_res = request.form
    _errlog =""

    if form_res['login'] != current_user.name :

        if len(form_res['login']) < 4 | len(form_res['login']) > 42:
            _errlog = "Votre pseudonyme est trop long ou trop court. -"

        checkuser1 = users.query.filter_by(name=form_res['login']).first()

        if not checkuser1 is None:
            _errlog += "Votre pseudonyme est deja utilisé. -"

    if form_res['email'] != current_user.email:
        if not re.match('[^@]+@[^@]+\.[^@]+', form_res['email']):
            _errlog += "Votre email n'est pas valide -"

        checkuser2 = users.query.filter_by(email=form_res['email']).first()
        if not checkuser2 is None:
            _errlog += 'Votre email est deja utilisé -'

    if form_res['mdp1'] is not None :
        if form_res['mdp1'] != form_res['mdp2']:
            _errlog += 'Les deux mots de passe ne sont pas identiques -'

        if len(form_res['mdp1']) < 4 | len(form_res['mdp1']) > 42:
            _errlog = "Votre mot de passe est trop long ou trop court. -"

        _mdp = sha256_crypt.encrypt(form_res['mdp1'])
    else:
        _mdp = current_user.pswd

    if _errlog != "":
        return render_template('pages/account' + LANG + '.html', err='true', errlog=_errlog, user=current_user)

    db.session.query(users).filter(users.id == current_user.id).update({users.name: form_res['login'],users.email: form_res['email'], users.pswd: _mdp},synchronize_session=False)
    db.session.commit()

    return redirect('/account')

@app.route('/accountcas')
def accountcas():
    LANG = getLang()
    user = users.query.filter_by(name=cas.username).first()

    if user is None:
        newuser = users(name=cas.username, email=cas.attributes['cas:email'], active=False, admin=False)
        db.session.add(newuser)
        db.session.commit()
        return render_template('pages/account' + LANG + '.html', access=False)

    if not user.active:
        return render_template('pages/account' + LANG + '.html', access=False)

    login_user(user, remember='remember')
    return account()


@app.route('/admin')
@login_required
def admin():
    LANG = getLang()
    if not current_user.admin:
        return account()

    _users = users.query.filter_by(admin=False).order_by(users.id)
    _reseaux = network.query.all()
    _capteurs = sensors.query.all()
    _mesures_var = measures_variable.query.all()

    return render_template('pages/admin' + LANG + '.html', users=_users, user=current_user.name, reseaux=_reseaux,
                           capteurs=_capteurs, mesures_var=_mesures_var)


@app.route('/deluser/<int:user_id>')
@login_required
def deluser(user_id):
    if not current_user.admin:
        return account()

    db.session.query(users).filter_by(id=user_id).delete()
    db.session.commit()

    _user = users.query.filter_by(id=user_id).first()

    msg = Message("Votre compte a été supprimé", recipients=[_user.email])
    msg.body = "L'administrateur du site EnviroDB à supprimer votre compte"
    mail.send(msg)

    return redirect('/admin')


@app.route('/upuser/<int:user_id>')
@login_required
def upuser(user_id):
    if not current_user.admin:
        return account()

    db.session.query(users).filter(users.id == user_id).update({users.active: True}, synchronize_session=False)
    db.session.commit()

    _user = users.query.filter_by(id=user_id).first()

    msg = Message("Votre compte a été activé", recipients=[_user.email])
    msg.body = "L'administrateur du site EnviroDB à accepté votre inscription"
    mail.send(msg)

    return redirect('/admin')


@app.route('/downuser/<int:user_id>')
@login_required
def downuser(user_id):
    if not current_user.admin:
        return account()

    db.session.query(users).filter(users.id == user_id).update({users.active: False}, synchronize_session=False)
    db.session.commit()

    _user = users.query.filter_by(id=user_id).first()

    msg = Message("Votre compte a été desactivé", recipients=[_user.email])
    msg.body = "L'administrateur du site EnviroDB à desactivé votre compte"
    mail.send(msg)

    return redirect('/admin')

@app.route('/resetkey/<int:user_id>')
@login_required
def resetkey(user_id):
    if not current_user.admin:
        return account()

    _user = users.query.filter_by(id=user_id).first()
    tokenCrypt = _user.encode_auth_token(_user.id)
    token = tokenCrypt.decode("utf-8")

    db.session.query(users).filter(users.id == user_id).update({users.token: token}, synchronize_session=False)
    db.session.commit()


    msg = Message("Votre clé api a été réinitialisé", recipients=[_user.email])
    msg.body = "L'administrateur du site EnviroDB à reinitialisé votre clé API"
    mail.send(msg)

    return redirect('/admin')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')


@app.route('/reqsearch', methods=['POST'])
@login_required
def reqsearch():
    LANG = getLang()
    form_res = request.form
    _value = form_res['value']
    _option = form_res['option']

    if _value is None:
        return redirect('/admin')

    _users = users.query.filter_by(admin=False).order_by(users.id)
    _reseaux = network.query.all()
    _capteurs = sensors.query.all()
    _mesures_var = measures_variable.query.all()

    res = []

    if _option == "mail":
        res = users.query.filter_by(email=_value).order_by(users.id)

    if _option == "name":
        res = users.query.filter_by(name=_value).order_by(users.id)

    return render_template('pages/admin' + LANG + '.html', users=_users, user=current_user.name, reseaux=_reseaux,
                           capteurs=_capteurs, mesures_var=_mesures_var, ressearch=res)


@app.route('/reqmod', methods=['POST'])
@login_required
def reqmod():
    form_res = request.form
    try:
        db.session.query(network).filter(network.id_network == form_res['id_network']).update(
            {network.name: form_res['name']}, synchronize_session=False)
        db.session.query(network).filter(network.id_network == form_res['id_network']).update(
            {network.descCourt: form_res['descCourt']}, synchronize_session=False)
        db.session.query(network).filter(network.id_network == form_res['id_network']).update(
            {network.descLong: form_res['descLong']}, synchronize_session=False)

        if (form_res['dateDebut'] != ""):
            db.session.query(network).filter(network.id_network == form_res['id_network']).update(
                {network.dateDebut: form_res['dateDebut']}, synchronize_session=False)

        if (form_res['dateFin'] != ""):
            db.session.query(network).filter(network.id_network == form_res['id_network']).update(
                {network.dateFin: form_res['dateFin']}, synchronize_session=False)

        db.session.commit()

        return redirect('/admin')
    except exc.SQLAlchemyError:
        flash("Un problème est survenu lors de l'ajout en base de la modification du réseau")
        return redirect('/admin')


@app.route('/reqsuppr/<int:id_reseau>')
@login_required
def reqsuppr(id_reseau):
    if not current_user.admin:
        return account()

    db.session.query(sensors).filter_by(id_network=id_reseau).delete()
    db.session.query(network).filter_by(id_network=id_reseau).delete()
    db.session.commit()

    return redirect('/admin')


@app.route('/reqaddNet', methods=['POST'])
@login_required
def reqaddNet():
    form_res = request.form
    _res = request.form.getlist('checkboxAdd' + form_res['user'])
    res = []
    for r in _res:
        res.append(int(r))

    try:
        con = connect(dbname = "dbuser",user = os.environ["POSTGRES_USER"],host = os.environ["POSTGRES_HOST"],password = os.environ["POSTGRES_PASSWORD"])
        cursor = con.cursor()
        cursor.execute("SELECT networkList FROM users WHERE users.id = " + form_res['user'])
        netList = cursor.fetchall()
        if type(netList[0]) is tuple:
            cursor.execute("UPDATE users SET networkList = %s WHERE users.id = %s", (res, form_res['user']))

        else:
            finalList = netList + res
            cursor.execute("UPDATE users SET networkList = %s WHERE users.id = %s", (finalList, form_res['user']))

        con.commit()
        cursor.close()
        con.close()

    except psycopg2.Error as e:
        flash("Un problème est survenu lors de l'ajout du droit au réseau de l'utilisateur" + e.pgerror)

    _user = users.query.filter_by(id=form_res['user']).first()
    msg = Message("Ajout de permissions", recipients=[_user.email])
    msg.body = "L'administrateur du site EnviroDB à autorisé l'accès aux réseaux"
    mail.send(msg)

    return redirect('/admin')


@app.route('/reqdelNet', methods=['POST'])
@login_required
def reqdelNet():
    form_res = request.form
    res = request.form.getlist('checkboxDel' + form_res['user'])
    try:
        con = connect(dbname = "dbuser",user = os.environ["POSTGRES_USER"],host = os.environ["POSTGRES_HOST"],password = os.environ["POSTGRES_PASSWORD"])

        cursor = con.cursor()
        cursor.execute("SELECT networkList FROM users WHERE users.id = " + form_res['user'])
        netList = db.cursor.fetchall()
        netList.remove(res)
        cursor.execute("UPDATE users SET users.networkList = ARRAY" + res + "WHERE users.id = " + form_res['user'])
        con.close()
        cursor.close()
    except psycopg2.Error as e:
        flash("Un problème est survenu lors de l'ajout du droit au réseau de l'utilisateur" + e.pgerror)

    _user = users.query.filter_by(id=form_res['user']).first()
    msg = Message("Suppression de permissions", recipients=[_user.email])
    msg.body = "L'administrateur du site EnviroDB à desactivé l'accès aux réseaux"
    mail.send(msg)
    return redirect('/admin')


@app.route('/reqaddRes', methods=['POST'])
@login_required
def reqaddRes():
    form_res = request.form
    try:
        now = datetime.now()
        reseau = network(name=form_res['name'], descCourt=form_res['descCourt'], descLong=form_res['descLong'],
                         dateDebut=now, dateFin=form_res['dateFin'])
        db.session.add(reseau)
        db.session.commit()
        return redirect('/admin')
    except exc.SQLAlchemyError:
        flash("Un problème est survenu lors de l'ajout en base du réseau")
        return redirect('/admin')


@app.route('/reqdelStation/<int:id_sensor>')
@login_required
def reqdelStation(id_sensor):
    if not current_user.admin:
        return account()

    db.session.query(sensors).filter_by(id_sensor=id_sensor).delete()
    db.session.commit()

    return redirect('/admin')


@app.route('/reqaddStation', methods=['POST'])
@login_required
def reqaddStation():
    form_res = request.form
    try:
        sensor = sensors(name=form_res['name'], latitude=form_res['latitude'], longitude=form_res['longitude'],
                         id_network=form_res['id_network'])
        db.session.add(sensor)
        db.session.commit()
        return redirect('/admin')
    except exc.SQLAlchemyError:
        flash("Un problème est survenu lors de l'ajout en base du capteur")
        return redirect('/admin')


@app.route('/reqmodStation', methods=['POST'])
@login_required
def reqmodStation():
    form_res = request.form
    try:
        db.session.query(sensors).filter(sensors.id_sensor == form_res['id_sensor']).update(
            {sensors.name: form_res['name']}, synchronize_session=False)
        db.session.query(sensors).filter(sensors.id_sensor == form_res['id_sensor']).update(
            {sensors.latitude: form_res['latitude']}, synchronize_session=False)
        db.session.query(sensors).filter(sensors.id_sensor == form_res['id_sensor']).update(
            {sensors.longitude: form_res['longitude']}, synchronize_session=False)
        db.session.commit()
        return redirect('/admin')
    except exc.SQLAlchemyError:
        flash("Un problème est survenu lors de l'ajout en base de la modification du capteur")
        return redirect('/admin')


@app.route('/reqaddStationCSV', methods=['POST'])
@login_required
def reqaddStationCSV():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect('/admin')
        if file:
            filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))

        form_res = request.form
    """_file = send_from_directory(UPLOAD_FOLDER,secure_filename(file.filename))"""
    return redirect('/admin')


@app.route('/reqdelMesure/<int:id_type>')
@login_required
def reqdelMesure(id_type):
    if not current_user.admin:
        return account()

    db.session.query(measures_variable).filter_by(id_type=id_type).delete()
    db.session.commit()

    return redirect('/admin')


@app.route('/reqaddVar', methods=['POST'])
@login_required
def reqaddVar():
    form_res = request.form
    if form_res['name'] is None or form_res['unite'] is None:
        flash("Un problème est survenu lors de l'ajout en base de la variable")
        return redirect('/admin')

    var = measures_variable(name=form_res['name'], unite=form_res['unite'], id_network=form_res['id_network'])
    db.session.add(var)
    db.session.commit()
    return redirect('/admin')


@app.route('/reqaddMesures', methods=['POST'])
@login_required
def reqaddMesures():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect('/admin')
        if file:
            filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))

    form_res = request.form
    return redirect('/admin')


@app.route('/reqselNetwork', methods=['POST'])
@login_required
def reqselNetwork():
    LANG = getLang()
    form_res = request.form
    return redirect('/networks/' + form_res['network'])


@app.route('/networks', defaults={'id_network': None})
@app.route('/networks/<int:id_network>')
@login_required
def networks(id_network):
    LANG = getLang()

    _reseaux = network.query.all()
    if id_network is None:
        return render_template('pages/network' + LANG + '.html', user=current_user, admin=current_user.admin,
                               networks=_reseaux, id_network="")

    _capteurs = sensors.query.filter_by(id_network=id_network)
    _network = network.query.filter_by(id_network=id_network).first()
    _mesures_var = measures_variable.query.filter_by(id_network=id_network)
    _sensor_params = sensors
    """
    capteurs = {}

    for capt in _capteurs:
        capteurs[capt.id_sensor] = {}
        capteurs[capt.id_sensor]['id_network'] = capt.id_network
        capteurs[capt.id_sensor]['id_sensor'] = capt.id_sensor
        capteurs[capt.id_sensor]['latitude'] = capt.latitude
        capteurs[capt.id_sensor]['longitude'] = capt.longitude
        capteurs[capt.id_sensor]['name'] = capt.name
   """

    con = connect(dbname = "envirodb",user = os.environ["POSTGRES_USER"],host = os.environ["POSTGRES_HOST"],password = os.environ["POSTGRES_PASSWORD"])

    cursor = con.cursor()

    """
    cursor.execute(
        "SELECT measures_variable.name,measures_variable.unite,network.id_network,sensors.id_sensor,sensors.latitude,sensors.longitude,sensors.name FROM measures_variable 
        INNER JOIN network on network.id_network = measures_variable.id_network 
        INNER JOIN sensors on network.id_network = sensors.id_network and network.id_network="
         + str(id_network) + " INNER JOIN measures on measures.id_station = sensors.id_sensor and measures.measure_type = measures_variable.id_type ")
    """


    cursor.execute(
        """SELECT sensors.id_sensor,sensors.latitude,sensors.longitude,sensors.name FROM sensors  
            where sensors.id_network=""" + str(id_network) )

    rows = cursor.fetchall()
 
    _capteurs = []
    for r in rows:
        cursor.execute("SELECT value, name FROM sensors_param WHERE id_sensor ="+str(r[0]))
        res = cursor.fetchall()
        paramsName = []
        paramsValue = []
        if (res):
            for r2 in res:
                paramsName.append(r2[1])
                paramsValue.append(r2[0])

        _capteurs.append({'latitude':r[1],'longitude':r[2],'name':r[3],'paramsName':paramsName,'paramsValue':paramsValue})



    """ TODO manage sensor by variable
    variables = {}
    for r in rows:
        cursor.execute("SELECT value, name FROM sensors_param WHERE id_sensor ="+str(r[3]))
        res = cursor.fetchall()
        paramsName = []
        paramsValue = []
        if (res):
            for r2 in res:
                paramsName.append(r2[1])
                paramsValue.append(r2[0])

        if r[0] not in variables.keys():
            variables[r[0]] = {}
            variables[r[0]]['unite'] = r[1]
            variables[r[0]][r[3]] = {}
            variables[r[0]][r[3]]['latitude'] = r[4]
            variables[r[0]][r[3]]['longitude'] = r[5]
            variables[r[0]][r[3]]['name'] = r[6]
            variables[r[0]][r[3]]['paramsName'] = paramsName
            variables[r[0]][r[3]]['paramsValue'] = paramsValue
        else:
            variables[r[0]][r[3]] = {}
            variables[r[0]][r[3]]['latitude'] = r[4]
            variables[r[0]][r[3]]['longitude'] = r[5]
            variables[r[0]][r[3]]['name'] = r[6]
            variables[r[0]][r[3]]['paramsName'] = paramsName
            variables[r[0]][r[3]]['paramsValue'] = paramsValue
    """
    con.close()
    cursor.close()
    return render_template('pages/network' + LANG + '.html', user=current_user, admin=current_user.admin,
                           id_network=id_network, network=_network, mesures_var=_mesures_var,
                           variables=_capteurs, networks=_reseaux)


@app.route('/measures',methods=['GET'])
def measures():
    LANG = getLang()
    con = connect(dbname = "envirodb",user = os.environ["POSTGRES_USER"],host = os.environ["POSTGRES_HOST"],password = os.environ["POSTGRES_PASSWORD"])

    cursor = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""select json_agg(children) from (select json_build_object( 'text', namenetwork, 'id_network', idnetwork, 'children', json_agg(children order by idnetwork)) as children from(
select namenetwork, idnetwork, json_build_object( 'text', namesensor, 'id_sensor', idsensor, 'children', json_agg(items order by idsensor)) as children from (
select namenetwork, idnetwork, namesensor, idsensor, json_build_object( 'id_var', idvar, 'text', namevar)as items from (
SELECT name AS namevar, var_id AS idvar, id_network AS idnetwork,network_name AS namenetwork, sensor_id AS idsensor, sensor_name as namesensor 
FROM jonction) 
AS sub group by idnetwork, idsensor, idvar, namenetwork, namesensor, namevar ) s 
group by idnetwork, idsensor, namenetwork, namesensor ) s 
group by idnetwork, namenetwork ) s;""")
    res1 = cursor.fetchall()

    con.close()
    cursor.close()

    tab_graph = []

    for i in range(MAX_GRAPH):
        cookie = request.cookies.get('graph'+str(i))
        if cookie is not None:
            tab_graph.append(request.cookies.get('graph'+str(i)))
        else:
            tab_graph.append('vide')

    if  tab_graph == ['vide'] * len(tab_graph):
        return render_template('pages/measures' + LANG + '.html',user=current_user, data=json.dumps(res1), graph=None, nbgraph=MAX_GRAPH)
    
    try:
        res2 = []
        for _graph in tab_graph :
            print(_graph)
            if _graph != 'vide':
                print(os.environ)
                #try:
                if True:
                    graph2 = _graph#.replace("\'", "\"")
                    jsondata = json.loads(graph2)
                    jsdata = jsondata['params']
                    dateStart = jsondata['interval'][0]
                    dateEnd = jsondata['interval'][1]
                    jsdata["data"] = []
                    jsdata["unite"] = []
                    jsdata["name"] = []
                    con = connect(dbname = "envirodb",user = os.environ["POSTGRES_USER"],host = os.environ["POSTGRES_HOST"],password = os.environ["POSTGRES_PASSWORD"])
                    cursor = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    for i in range(0, len(jsdata['idvar'])):
                        cursor.execute("SELECT measures.date, measure FROM measures WHERE measures.measure_type =" + str(
                            jsdata["idvar"][i]) + "AND id_station = " + str(
                            jsdata["idsensor"][i]) + " AND measures.date >= timestamp '" + str(
                            dateStart) + "' AND measures.date < timestamp '" + str(dateEnd) + "' ORDER BY measures.date")
                        res = cursor.fetchall()
                        cursor.execute("SELECT measures_variable.unite FROM measures_variable WHERE measures_variable.id_type ="+str(jsdata["idvar"][i]))
                        params = cursor.fetchone()
                        _res2 = dict(params)
                        jsdata["unite"].append(_res2["unite"])
                        _res = [dict(row) for row in res]
                        jsdata["data"].append(_res)
                        for y in range(0, len(jsdata['data'][i])):
                            jsdata["data"][i][y]["date"] = str(jsdata['data'][i][y]['date'])

                    cursor.close()
                    con.close()
                    res2.append(json.dumps(jsdata))
                else:
                    pass

            else :
                res2.append(("vide"))
        return render_template('pages/measures' + LANG + '.html', user=current_user, data=json.dumps(res1), graph=res2, nbgraph=MAX_GRAPH)
    except exc.SQLAlchemyError:
        return render_template('pages/measures' + LANG + '.html', user=current_user, data=json.dumps(res1), graph=res2, nbgraph=MAX_GRAPH, err=True, errlog="Error while getting graph data")



@app.route('/ajaxpostmethod', methods=['POST'])
def ajaxpostmethod():
    dateStart = request.form['dateStart']
    dateEnd = request.form['dateEnd']

    cookie = {}
    cookie["params"] = json.loads(request.form['paramsList'])
    cookie["interval"] = [dateStart, dateEnd]
    resp = make_response()
    name = 'graph' + request.form['id']
    resp.set_cookie(name, json.dumps(cookie))

    return resp

@app.route('/reqexport',methods=['POST'])
@login_required
def reqexport():
    data = request.form['data']
    id = request.form['id']
    dataDict = eval(data)
    fileDict=[]
    for i in range(len(dataDict['idvar'])):
        measureDict = {}
        measureDict['network'] = str(dataDict['namenetwork'][i])
        measureDict['sensor'] = str(dataDict['namesensor'][i])
        measureDict['variable'] = str(dataDict['namesvar'][i])
        measureDict['unit'] = str(dataDict['unite'][i])
        measureDict['data'] = str(dataDict['data'][i])
        fileDict.append(measureDict)
    print('fileDict:',fileDict)
    respFile=json.dumps(fileDict, ensure_ascii=False).encode('utf8')

    return Response(respFile,
            mimetype='application/json',
            headers={'Content-Disposition':'attachment;filename=graph'+str(id)+'.json'})

    
