"""
    Flask-based server for the Robo-AO Data Archive

    Dr Dmitry A. Duev @ Caltech, 2016-2017
"""

import matplotlib
matplotlib.use('Agg')

import ast
import os
import pymongo
import json
import datetime
import pytz
import argparse
import sys
import subprocess
import inspect
from collections import OrderedDict
import flask
import flask_login
from werkzeug.security import generate_password_hash, check_password_hash
from urllib.parse import urlparse
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.io import fits
import astropy.units as u
import pyvo as vo
from PIL import Image
from urllib.request import urlopen
from io import StringIO, BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import aplpy
import base64
import shutil


class ReverseProxied(object):
    """
    Wrap the application in this middleware and configure the
    front-end server to add these headers, to let you quietly bind
    this to a URL other than / and to an HTTP scheme that is
    different than what is used locally.

    In nginx:
    location /myprefix {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Scheme $scheme;
        proxy_set_header X-Script-Name /myprefix;
        }

    :param app: the WSGI application
    """
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        script_name = environ.get('HTTP_X_SCRIPT_NAME', '')
        if script_name:
            environ['SCRIPT_NAME'] = script_name
            path_info = environ['PATH_INFO']
            if path_info.startswith(script_name):
                environ['PATH_INFO'] = path_info[len(script_name):]

        scheme = environ.get('HTTP_X_SCHEME', '')
        if scheme:
            environ['wsgi.url_scheme'] = scheme

        server = environ.get('HTTP_X_FORWARDED_SERVER', '')
        if server:
            environ['HTTP_HOST'] = server

        return self.app(environ, start_response)


def utc_now():
    return datetime.datetime.now(pytz.utc)


def get_filter_code(_filter):
    # print(_filter)
    if _filter == 'Sloan g':
        return 'Sg'
    elif _filter == 'Sloan r':
        return 'Sr'
    elif _filter == 'Sloan i':
        return 'Si'
    elif _filter == 'Sloan z':
        return 'Sz'
    elif _filter == 'Longpass 600nm':
        return 'lp600'
    elif _filter == 'Clear':
        return 'c'
    # elif _filter == 'H':
    #     return 'H'
    elif _filter == 'J':
        return 'J'
    else:
        raise Exception('couldn\'t recognize filter name')


def radec_str2rad(_ra_str, _dec_str):
    """

    :param _ra_str: 'H:M:S'
    :param _dec_str: 'D:M:S'
    :return: ra, dec in rad
    """
    # convert to rad:
    _ra = list(map(float, _ra_str.split(':')))
    _ra = (_ra[0] + _ra[1] / 60.0 + _ra[2] / 3600.0) * np.pi / 12.
    _dec = list(map(float, _dec_str.split(':')))
    _dec = (_dec[0] + _dec[1] / 60.0 + _dec[2] / 3600.0) * np.pi / 180.

    return _ra, _dec


def great_circle_distance(phi1, lambda1, phi2, lambda2):
    # input: dec1, ra1, dec2, ra2 [rad]
    # this is much faster than astropy.coordinates.Skycoord.separation
    delta_lambda = np.abs(lambda2 - lambda1)
    return np.arctan2(np.sqrt((np.cos(phi2)*np.sin(delta_lambda))**2
                              + (np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(delta_lambda))**2),
                      np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(delta_lambda))


def get_config(_config_file='config.json'):
    """
        load config data
    """
    try:
        ''' script absolute location '''
        abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

        if _config_file[0] not in ('/', '~'):
            if os.path.isfile(os.path.join(abs_path, _config_file)):
                config_path = os.path.join(abs_path, _config_file)
            else:
                raise IOError('Failed to find config file')
        else:
            if os.path.isfile(_config_file):
                config_path = _config_file
            else:
                raise IOError('Failed to find config file')

        with open(config_path) as cjson:
            config_data = json.load(cjson)
            # config must not be empty:
            if len(config_data) > 0:
                return config_data
            else:
                raise Exception('Failed to load config file')

    except Exception as _e:
        print(_e)
        raise Exception('Failed to read in the config file')


def parse_obs_name(_obs, _program_pi):
    """
        Parse Robo-AO observation name
    :param _obs:
    :param _program_pi: dict program_num -> PI
    :return:
    """
    # parse name:
    _tmp = _obs.split('_')
    # program num. it will be a string in the future
    _prog_num = str(_tmp[0])
    # who's pi?
    if _prog_num in _program_pi.keys():
        _prog_pi = _program_pi[_prog_num]
    else:
        # play safe if pi's unknown:
        _prog_pi = ['admin']
    # stack name together if necessary (if contains underscores):
    _sou_name = '_'.join(_tmp[1:-5])
    # code of the filter used:
    _filt = _tmp[-4:-3][0]
    # date and time of obs:
    _date_utc = datetime.datetime.strptime(_tmp[-2] + _tmp[-1], '%Y%m%d%H%M%S.%f')
    # camera:
    _camera = _tmp[-5:-4][0]
    # marker:
    _marker = _tmp[-3:-2][0]

    return _prog_num, _prog_pi, _sou_name, _filt, _date_utc, _camera, _marker


def connect_to_db(_config):
    """ Connect to the mongodb database

    :return:
    """
    try:
        if _config['server']['environment'] == 'production':
            # in production, must set up replica set
            _client = pymongo.MongoClient(host=_config['database']['host'], port=_config['database']['port'],
                                          replicaset=_config['database']['replicaset'],
                                          readPreference='primaryPreferred')
        else:
            # standalone from my laptop, when there's only one instance of DB
            _client = pymongo.MongoClient(host=_config['database']['host'], port=_config['database']['port'])
        # grab main database:
        _db = _client[_config['database']['db']]
    except Exception as _e:
        raise ConnectionRefusedError
    try:
        # authenticate
        _db.authenticate(_config['database']['user'], _config['database']['pwd'])
    except Exception as _e:
        raise ConnectionRefusedError
    try:
        # get collection with observations
        _coll_obs = _db[_config['database']['collection_obs']]
    except Exception as _e:
        raise NameError
    try:
        # get collection with auxiliary stuff
        _coll_aux = _db[_config['database']['collection_aux']]
    except Exception as _e:
        raise NameError

    _coll_weather = None

    try:
        # get collection with user access credentials
        _coll_usr = _db[_config['database']['collection_pwd']]
    except Exception as _e:
        raise NameError
    try:
        # build dictionary program num -> pi name
        cursor = _coll_usr.find()
        _program_pi = {}
        for doc in cursor:
            # handle admin separately
            if doc['_id'] == 'admin':
                continue
            _progs = doc['programs']
            for v in _progs:
                # multiple users could have access to the same program, that's totally fine!
                if str(v) not in _program_pi:
                    _program_pi[str(v)] = [doc['_id'].encode('ascii', 'ignore')]
                else:
                    _program_pi[str(v)].append(doc['_id'].encode('ascii', 'ignore'))
                    # print(program_pi)
    except Exception as _e:
        _program_pi = {}

    return _client, _db, _coll_obs, _coll_usr, _coll_aux, _coll_weather, _program_pi


''' initialize the Flask app '''
app = flask.Flask(__name__)
app.wsgi_app = ReverseProxied(app.wsgi_app)
app.secret_key = 'roboaokicksass'
# add 'do' statement to jinja environment (does the same as {{ }}, but return nothing):
app.jinja_env.add_extension('jinja2.ext.do')


def get_db(_config):
    """Opens a new database connection if there is none yet for the
    current application context.
    """
    if not hasattr(flask.g, 'client'):
        flask.g.client, flask.g.db, flask.g.coll, flask.g.coll_usr, \
        flask.g.coll_aux, flask.g.coll_weather, flask.g.program_pi = connect_to_db(_config)
    return flask.g.client, flask.g.db, flask.g.coll, flask.g.coll_usr, \
                flask.g.coll_aux, flask.g.coll_weather, flask.g.program_pi


@app.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    if hasattr(flask.g, 'client'):
        flask.g.client.close()


login_manager = flask_login.LoginManager()

login_manager.init_app(app)


''' Create command line argument parser if run from command line in test environment '''
# FIXME:
env = 'production'
# env = 'test'

if env != 'production':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Data archive for Robo-AO')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)

    args = parser.parse_args()
    config_file = args.config_file
else:
    # FIXME:
    config_file = 'config.archive.json'
    # config_file = 'config.json'
    # config_file = 'config.analysis.json'


''' get config data '''
try:
    config = get_config(_config_file=config_file)
except IOError:
    config = get_config(_config_file='config.json')
# print(config)

''' serve additional static data (preview images, compressed source data) '''
# this is done with flask's send_from_directory() function due to security concerns
# in case if static files are served with


@app.route('/data/<path:filename>')
@flask_login.login_required
def data_static(filename):
    """
        Get files from the archive
    :param filename:
    :return:
    """
    _p, _f = os.path.split(filename)
    return flask.send_from_directory(os.path.join(config['path']['path_archive'], _p), _f)


''' handle user login'''


class User(flask_login.UserMixin):
    pass


@login_manager.user_loader
def user_loader(username):
    # look up username in database:
    client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
    select = coll_usr.find_one({'_id': username})
    if select is None:
        return

    user = User()
    user.id = username
    return user


@login_manager.request_loader
def request_loader(request):
    username = request.form.get('username')
    client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
    # look up in the database
    select = coll_usr.find_one({'_id': username})
    if select is None:
        return

    user = User()
    user.id = username

    try:
        user.is_authenticated = check_password_hash(select['password'],
                                                    flask.request.form['password'])
    except Exception as _e:
        print(_e)
        return

    return user


@app.route('/login', methods=['GET', 'POST'])
def login():
    if flask.request.method == 'GET':
        # logged in already?
        if flask_login.current_user.is_authenticated:
            return flask.redirect(flask.url_for('root'))
        # serve template if not:
        else:
            return flask.render_template('template-login.html', fail=False, current_year=datetime.datetime.now().year)
    # print(flask.request.form['username'], flask.request.form['password'])

    username = flask.request.form['username']
    # check if username exists and passwords match
    # look up in the database first:
    client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
    select = coll_usr.find_one({'_id': username})
    if select is not None and \
            check_password_hash(select['password'], flask.request.form['password']):
        user = User()
        user.id = username
        flask_login.login_user(user, remember=True)
        return flask.redirect(flask.url_for('root'))
    else:
        # serve template with flag fail=True to display fail message
        return flask.render_template('template-login.html', fail=True, current_year=datetime.datetime.now().year)


def stream_template(template_name, **context):
    """
        see: http://flask.pocoo.org/docs/0.11/patterns/streaming/
    :param template_name:
    :param context:
    :return:
    """
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.enable_buffering(5)
    return rv


def get_dates(user_id, coll, coll_aux, start=None, stop=None, max_dates=100):
    """
        Get science and auxiliary data from start to stop
    :param user_id:
    :param coll:
    :param coll_aux:
    :param start:
    :param stop:
    :param max_dates
    :return:
    """
    if start is None:
        # this is ~when we moved to KP:
        start = datetime.datetime(2015, 10, 1)
        # get the last date with observations:
        max_dates = 1
        # or last N days:
        # start = datetime.datetime.utcnow() - datetime.timedelta(days=10)
    else:
        try:
            start = datetime.datetime.strptime(start, '%Y%m%d')
        except Exception as _e:
            print(_e)
            # get the last date with observations:
            max_dates = 1
            start = datetime.datetime(2015, 10, 1)
            # start = datetime.datetime.utcnow() - datetime.timedelta(days=1)

    if stop is None:
        stop = datetime.datetime.utcnow()
    else:
        try:
            stop = datetime.datetime.strptime(stop, '%Y%m%d')
            # if stop < start:
            #     stop = datetime.datetime.utcnow()
        except Exception as _e:
            print(_e)
            stop = datetime.datetime.utcnow()

    # provide some feedback!
    messages = []

    # create indices not to perform in-memory sorting:
    try:
        coll.create_index([('date_utc', -1)])
        coll_aux.create_index([('_id', 1)])
    except Exception as _e:
        print(_e)
        message = (_e, 'error')
        messages.append(message)

    # dictionary: {date: {data: {program_N: [observations]}, aux: {}}}
    dates = dict()

    if user_id == 'admin':
        # get everything;
        cursor = coll.find({'date_utc': {'$gte': start, '$lt': stop}})
    else:
        # get only programs accessible to this user marked as distributed:
        cursor = coll.find({'date_utc': {'$gte': start, '$lt': stop},
                            'science_program.program_PI': user_id,
                            'distributed.status': True})

    # iterate over query result for science data:
    try:
        for obs in cursor.sort([('date_utc', -1)]):
            date = obs['date_utc'].strftime('%Y%m%d')
            # add key to dict if it is not there already:
            if date not in dates:
                if len(dates) + 1 > max_dates:
                    # this is to prevent trying to load everything stored on the server
                    # and reduce traffic in general. Do not show that if max_dates == 1, i.e. on start page:
                    if max_dates != 1:
                        message = ('Too much data requested, showing last {:d} dates only'.format(max_dates), 'warning')
                        messages.append(message)
                    break
                dates[date] = {'data': {}, 'aux': {}}
            # add key for program if it is not there yet
            program_id = obs['science_program']['program_id']
            if program_id not in dates[date]['data']:
                dates[date]['data'][program_id] = []
            dates[date]['data'][program_id].append(obs)
    except Exception as _e:
        print(_e)
        message = (_e, 'error')
        messages.append(message)

    # get aux data
    # cursor_aux = coll_aux.find({'_id': {'$gte': start.strftime('%Y%m%d'), '$lt': stop.strftime('%Y%m%d')}})
    start_aux = sorted(dates.keys())[0] if len(dates) > 0 else start.strftime('%Y%m%d')
    cursor_aux = coll_aux.find({'_id': {'$gte': start_aux, '$lt': stop.strftime('%Y%m%d')}})

    # iterate over query result for aux data:
    try:
        for date_data in cursor_aux:
            # string date
            date = date_data['_id']

            aux = OrderedDict()

            for key in ('seeing', 'contrast_curve', 'strehl', 'calib'):
                if key in date_data and (key == 'calib' or date_data[key]['done']):
                    aux[key] = dict()
                    aux[key]['done'] = True if (key in date_data and date_data[key]['done']) else False
                    if key == 'seeing':
                        # for seeing data, fetch frame names to show in a 'movie'
                        aux[key]['frames'] = []
                        # sort by time, not by name:
                        # don't consider failed frames
                        seeing_frames = [frame for frame in date_data[key]['frames'] if None not in frame]
                        ind_sort = np.argsort([frame[1] for frame in seeing_frames])
                        for frame in np.array(seeing_frames)[ind_sort]:
                            if not np.any(np.equal(frame, None)):
                                aux[key]['frames'].append(frame[0] + '.png')

            if len(aux) > 0:
                # show aux data for the admin all the time
                if user_id == 'admin':
                    # init entry if no science
                    if date not in dates:
                        dates[date] = {'data': {}, 'aux': {}}
                    # add to dates:
                    dates[date]['aux'] = aux
                # for regular users, only show aux data for dates with obs of user's programs
                else:
                    if date in dates:
                        dates[date]['aux'] = aux
    except Exception as _e:
        print(_e)
        message = (_e, 'error')
        messages.append(message)

    # print(dates)
    # latest obs - first
    # dates = sorted(list(set(dates)), reverse=True)

    return dates, messages


def get_dates_psflib(coll_aux, start=None, stop=None):
    """
        Get science and auxiliary data from start to stop
    :param coll_aux:
    :param start:
    :param stop:
    :return:
    """
    if start is None:
        # this is ~when we moved to KP:
        # start = datetime.datetime(2015, 10, 1)
        # by default -- last 30 days:
        start = datetime.datetime.utcnow() - datetime.timedelta(days=10)
    else:
        try:
            start = datetime.datetime.strptime(start, '%Y%m%d')
        except Exception as _e:
            print(_e)
            start = datetime.datetime.utcnow() - datetime.timedelta(days=10)

    if stop is None:
        stop = datetime.datetime.utcnow()
    else:
        try:
            stop = datetime.datetime.strptime(stop, '%Y%m%d')
            # if stop < start:
            #     stop = datetime.datetime.utcnow()
        except Exception as _e:
            print(_e)
            stop = datetime.datetime.utcnow()

    # create indices not to perform in-memory sorting:
    try:
        coll_aux.create_index([('_id', 1)])
    except Exception as _e:
        print(_e)

    # dictionary: {date: {ob_id: {data}}}
    dates = dict()

    # get aux data
    cursor_aux = coll_aux.find({'_id': {'$gte': start.strftime('%Y%m%d'), '$lt': stop.strftime('%Y%m%d')}})

    # iterate over query result for aux data:
    try:
        for date_data in cursor_aux:
            # string date
            date = date_data['_id']

            psfdata = OrderedDict()

            if 'psf_lib' in date_data:
                for _obs in date_data['psf_lib']:
                    psfdata[_obs] = date_data['psf_lib'][_obs]

            if len(psfdata) > 0:
                dates[date] = psfdata
    except Exception as _e:
        print(_e)

    return dates


@app.route('/_get_fits_header')
@flask_login.login_required
def get_fits_header():
    """
        Get FITS header for a _source_ observed on a _date_
    :return: jsonified dictionary with the header / empty dict if failed
    """
    user_id = flask_login.current_user.id

    # get parameters from the AJAX GET request
    _obs = flask.request.args.get('source', 0, type=str)

    _, _, coll, _, _, _, _program_pi = get_db(config)

    # trying to steal stuff?
    _program, _, _, _, _, _, _ = parse_obs_name(_obs, _program_pi)
    if (user_id != 'admin') and (user_id not in _program_pi[_program]):
        # flask.abort(403)
        return flask.jsonify(result={})

    cursor = coll.find({'_id': _obs})

    try:
        if cursor.count() == 1:
            for obs in cursor:
                header = obs['fits_header']
            return flask.jsonify(result=OrderedDict(header))
        # not found in the database?
        else:
            return flask.jsonify(result={})
    except Exception as _e:
        print(_e)
        return flask.jsonify(result={})


@app.route('/_get_vo_image', methods=['GET'])
@flask_login.login_required
def get_vo_image():
    user_id = flask_login.current_user.id

    # get parameters from the AJAX GET request
    _obs = flask.request.args.get('source', 0, type=str)

    _, _, coll, _, _, _, _program_pi = get_db(config)

    # trying to steal stuff?
    _program, _, _, _, _, _, _ = parse_obs_name(_obs, _program_pi)
    if (user_id != 'admin') and (user_id not in _program_pi[_program]):
        # flask.abort(403)
        return flask.jsonify(result={})

    # print(_obs)
    cursor = coll.find({'_id': _obs})

    try:
        if cursor.count() == 1:
            for obs in cursor:
                header = obs['fits_header']
                if 'TELDEC' in header and 'TELRA' in header:
                    c = SkyCoord(header['TELRA'][0], header['TELDEC'][0],
                                 unit=(units.hourangle, units.deg), frame='icrs')
                    # print(c)
                    # print(c.ra.deg, c.dec.deg)

                    output = {}

                    # PS1 i'-band image:
                    image_names_url = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?' + \
                                      'ra={:f}&dec={:f}&filters=i'.format(c.ra.deg, c.dec.deg)

                    # print(image_names_url)

                    response = urlopen(image_names_url).readlines()[-1]
                    # print(response)
                    fits_location = response.split()[-2]
                    fits_name = response.split()[-1]
                    # print(fits_location, fits_name)

                    # 36"x36"
                    size_pix = 144
                    # 10"x10"
                    # size_pix = 40

                    fits_cutout = 'http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?' + \
                                  'red={:s}&format=fits&x={:f}&y={:f}&size={:d}'.format(str(fits_location)[2:-1],
                                                                                        c.ra.deg,
                                                                                        c.dec.deg,
                                                                                        size_pix) + \
                                  '&wcs=1&imagename={:s}'.format(str(fits_name)[2:-1])

                    # print(fits_cutout)
                    # print(urlopen(fits_cutout).read())

                    figure = matplotlib.figure.Figure()
                    canvas = FigureCanvas(figure)

                    with fits.open(fits_cutout) as ps_fits:
                        fig = aplpy.FITSFigure(data=ps_fits, figure=figure)
                        fig.show_colorscale(cmap='magma')

                    fig.add_grid()
                    fig.grid.show()
                    fig.grid.set_color('gray')
                    fig.grid.set_alpha(0.85)
                    fig.grid.set_linewidth(0.7)

                    # add scale bar
                    scale_bar_size = 5  # "
                    fig.add_scalebar(length=scale_bar_size * u.arcsecond)
                    fig.scalebar.set_alpha(0.7)
                    fig.scalebar.set_color('white')
                    fig.scalebar.set_label('{:d}\"'.format(scale_bar_size))

                    png_output = BytesIO()
                    # png_output = StringIO()
                    figure.savefig(png_output, format='png')
                    # canvas.print_png(png_output)
                    # png_output = str(png_output.getvalue())
                    png_output = png_output.getvalue()
                    png_output = str(base64.b64encode(png_output))[2:-1]

                    output['{:s}_{:s}'.format('ps1', 'i')] = png_output

                    if False:
                        vo_url = config['misc']['vo_server']

                        survey_filter = {'dss2': 'r', '2mass': 'h'}

                        for survey in survey_filter:

                            # TODO: use image size + scale from config. For now it's 36"x36" times 2 (0.02 deg^2)
                            previews = vo.imagesearch(vo_url, pos=(c.ra.deg, c.dec.deg),
                                                      size=(0.01*2, 0.01*2), format='image/png',
                                                      survey=survey)
                            if previews.nrecs == 0:
                                continue
                            # get url:
                            image_url = [image.getdataurl() for image in previews
                                         if image.title == survey + survey_filter[survey]][0]

                            # print(image_url)

                            _file = StringIO(urlopen(image_url).read())
                            survey_image = np.array(Image.open(_file))

                            fig = plt.figure(figsize=(3, 3))
                            ax = fig.add_subplot(111)
                            ax.imshow(survey_image, origin='upper', interpolation='nearest', cmap=plt.cm.magma)
                            ax.grid('off')
                            ax.set_axis_off()
                            fig.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, right=1, left=0)
                            plt.margins(0, 0)
                            # flip x axis for it to look like our images:
                            plt.gca().invert_xaxis()
                            plt.gca().invert_yaxis()

                            canvas = FigureCanvas(fig)
                            png_output = StringIO()
                            canvas.print_png(png_output)
                            png_output = png_output.getvalue().encode('base64')

                            output['{:s}_{:s}'.format(survey, survey_filter[survey])] = png_output

            return flask.jsonify(result=output)
        # not found in the database?
        else:
            return flask.jsonify(result={})
    except Exception as _e:
        print(_e)
        return flask.jsonify(result={})


@app.route('/_force_redo')
@flask_login.login_required
def force_redo():
    """
        Force redo a pipeline for _source_
    :return: jsonified dictionary with the header / empty dict if failed
    """
    user_id = flask_login.current_user.id

    # get parameters from the AJAX GET request
    _obs = flask.request.args.get('source', 0, type=str)
    _pipe = flask.request.args.get('pipe', 1, type=str)

    if _pipe not in ('bright_star.pca', 'bright_star.strehl', 'faint_star', 'faint_star.pca', 'faint_star.strehl'):
        return flask.jsonify(result={'success': False, 'status': 'bad request'})

    _, _, _coll, _, _, _, _program_pi = get_db(config)

    # trying to steal stuff? no toys for bad boys!
    _program, _, _, _, _, _, _ = parse_obs_name(_obs, _program_pi)
    if (user_id != 'admin') and (user_id not in _program_pi[_program]):
        # flask.abort(403)
        return flask.jsonify(result={'success': False, 'status': 'not yours!'})

    try:
        # check that it's not being redone at the moment already:
        go_ahead = False

        _o = _coll.find_one({'_id': _obs})
        if _pipe == 'faint_star':
            go_ahead = not (_o['pipelined']['faint_star']['status']['force_redo'] or
                            _o['pipelined']['faint_star']['status']['enqueued'])
        else:
            for _p in ('bright_star', 'faint_star'):
                if _pipe == '{:s}.pca'.format(_p):
                    go_ahead = not (_o['pipelined'][_p]['pca']['status']['force_redo'] or
                                    _o['pipelined'][_p]['pca']['status']['enqueued'])
                elif _pipe == '{:s}.strehl'.format(_p):
                    go_ahead = not (_o['pipelined'][_p]['strehl']['status']['force_redo'] or
                                    _o['pipelined'][_p]['strehl']['status']['enqueued'])

        # set corresponding force_redo flag:
        if go_ahead:
            _status = _coll.update_one(
                {'_id': _obs},
                {
                    '$set': {
                        'pipelined.{:s}.status.force_redo'.format(_pipe): True,
                    }
                }
            )
            if _status.matched_count == 1:
                return flask.jsonify(result={'success': True, 'status': 'successfully enqueued'})
            # not found in the database?
            else:
                return flask.jsonify(result={'success': False, 'status': 'could not enqueue'})
        else:
            return flask.jsonify(result={'success': False, 'status': 'already enqueued'})
    except Exception as _e:
        print(_e)
        return flask.jsonify(result={'success': False, 'status': 'failed to enqueue'})


@app.route('/_copy_calib')
@flask_login.login_required
def copy_calib():
    """
        Copy master calibration files for a date
    :return: jsonified dictionary: result: success or failure
    """
    user_id = flask_login.current_user.id

    if user_id != 'admin':
        # flask.abort(403)
        return flask.jsonify(result={'success': False, 'status': 'You are not an admin!'})

    # get parameters from the AJAX GET request
    _date = flask.request.args.get('date', 0, type=str)

    _, _, _, _, _coll_aux, _, _program_pi = get_db(config)

    try:

        _select = _coll_aux.find({'calib.done': True}, {'_id': 1})

        _d0 = datetime.datetime.strptime(_date, '%Y%m%d')

        _dates = []
        for _doc in _select:
            _dates.append(datetime.datetime.strptime(_doc['_id'], '%Y%m%d'))

        _dates = np.array(_dates)

        nearest_date = _dates[np.argmin([np.abs((_d - _d0).total_seconds()) for _d in _dates])].strftime('%Y%m%d')
        # print(nearest_date)

        # make calib dir:
        _path_out = os.path.join(config['path']['path_archive'], _date, 'calib')
        # output dir exists?
        if not os.path.exists(_path_out):
            os.makedirs(_path_out)

        _path_in = os.path.join(config['path']['path_archive'], nearest_date, 'calib')

        # copy master calib files over:
        for _f in os.listdir(_path_in):
            shutil.copy2(os.path.join(_path_in, _f), _path_out)

        # rsync archive and analysis machines
        if config['server']['environment'] == 'production':
            try:
                subprocess.run(["/usr/local/bin/rsync", "-av", "-e", "'ssh -p 22220'",
                                _path_out + '/', "roboao@140.252.53.120:/Data1/archive/{:s}/calib/".format(_date),
                                ">/dev/null", "2>&1"])
            except Exception as _e:
                print(_e)
                return flask.jsonify(result={'success': False,
                                             'status': 'could not rsync analysis and archive machines'})

        # mark calib as done in the DB:
        _status = _coll_aux.update_one(
            {'_id': _date},
            {
                '$set': {
                    'calib.done': True,
                    'calib.last_modified': utc_now(),
                    'calib.comment': 'Copied from {:s}'.format(nearest_date)
                }
            }
        )

        if _status.matched_count == 1:
        # if 1:
            return flask.jsonify(result={'success': True, 'status': 'successfully copied'})
        # not found in the database?
        else:
            return flask.jsonify(result={'success': False, 'status': 'could not find date'})

    except Exception as _e:
        print(_e)
        return flask.jsonify(result={'success': False, 'status': 'failed to copy calib data'})


@app.route('/_enqueue_lib')
@flask_login.login_required
def enqueue_lib():
    """
    """
    try:
        user_id = flask_login.current_user.id

        if user_id == 'admin':
            # get parameters from the AJAX GET request
            _date = flask.request.args.get('date', 0, type=str)
            _obs = flask.request.args.get('source', 1, type=str)
            _action = flask.request.args.get('action', 2, type=str)

            assert _action in ('add', 'remove'), 'bad action requested'
            if _action == 'add':
                _status = 'add_to_lib'
            elif _action == 'remove':
                _status = 'remove_from_lib'

            _, _, _, _, _coll_aux, _, _ = get_db(config)

            _d = _coll_aux.find_one({'_id': _date})
            assert _obs in _d['psf_lib'], 'could not find {:s} in {:s}'.format(_obs, _date)

            # check that it's not being redone at the moment already:
            go_ahead = not _d['psf_lib'][_obs]['enqueued']

            if go_ahead:
                _status = _coll_aux.update_one(
                    {'_id': _date},
                    {
                        '$set': {
                            'psf_lib.{:s}.enqueued'.format(_obs): True,
                            'psf_lib.{:s}.status'.format(_obs): _status
                        }
                    }
                )
                if _status.matched_count == 1:
                    return flask.jsonify(result={'success': True, 'status': 'successfully enqueued'})
                # not found in the database?
                else:
                    return flask.jsonify(result={'success': False, 'status': 'could not enqueue'})
            else:
                return flask.jsonify(result={'success': False, 'status': 'already enqueued'})

        else:
            return flask.jsonify(result={'success': False, 'status': 'not an admin'})

    except Exception as _e:
        print(_e)
        return flask.jsonify(result={'success': False, 'status': 'failed to enqueue'})


@app.route('/_rerun_pca_date')
@flask_login.login_required
def rerun_pca_date():
    """
    """
    try:
        user_id = flask_login.current_user.id

        if user_id == 'admin':
            # get parameters from the AJAX GET request
            _date = flask.request.args.get('date', 0, type=str)

            start = datetime.datetime.strptime(_date, '%Y%m%d')
            start = start.replace(tzinfo=pytz.utc)
            stop = start + datetime.timedelta(days=1)

            _, _, _coll, _, _, _, _ = get_db(config)

            query = dict()
            query['date_utc'] = {'$gte': start, '$lt': stop}
            query['pipelined.bright_star.pca.status.done'] = {'$eq': True}

            _status = _coll.update_many(
                query,
                {
                    '$set': {
                        'pipelined.bright_star.pca.status.force_redo': True
                    }
                }
            )

            return flask.jsonify(result={'success': True, 'status': 'will force redo!'})

        else:
            return flask.jsonify(result={'success': False, 'status': 'not an admin'})

    except Exception as _e:
        print(_e)
        return flask.jsonify(result={'success': False, 'status': 'failed to enqueue'})


@app.route('/get_data', methods=['GET'])
@flask_login.login_required
def wget_script():
    """
        Generate bash script to fetch all data for date/program with wget
    :return:
    """
    # get url
    url_parsed = urlparse(flask.request.url)
    path_parsed = os.path.split(url_parsed.path)
    if path_parsed[0] != '/':
        url = url_parsed.netloc + path_parsed[0]
    else:
        url = url_parsed.netloc

    _date_str = flask.request.args['date']
    _date = datetime.datetime.strptime(_date_str, '%Y%m%d')
    _program = flask.request.args['program']

    user_id = flask_login.current_user.id
    # get db connection
    client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)

    # trying to get something you're not supposed to get?
    if user_id != 'admin' and program_pi[_program] != user_id:
        flask.abort(403)
    else:
        cursor = coll.find({'date_utc': {'$gte': _date,
                                         '$lt': _date + datetime.timedelta(days=1)},
                            'science_program.program_id': _program,
                            'distributed.status': True})
        response_text = '#!/usr/bin/env bash\n'
        for obs in cursor:
            response_text += 'wget http://{:s}/data/{:s}/{:s}/{:s}.tar.bz2\n'.format(url, _date_str,
                                                                                obs['_id'], obs['_id'])
        # print(response_text)

        # generate .sh file on the fly
        response = flask.make_response(response_text)
        response.headers['Content-Disposition'] = \
            'attachment; filename=program_{:s}_{:s}.wget.sh'.format(_program, _date_str)
        return response


@app.route('/get_data_by_id', methods=['POST'])
@flask_login.login_required
def wget_script_by_id():
    """
        Generate bash script to fetch data by id with wget
    :return:
    """
    # print(flask.request.url)
    url_parsed = urlparse(flask.request.url)
    path_parsed = os.path.split(url_parsed.path)
    if path_parsed[0] != '/':
        url = url_parsed.netloc + path_parsed[0]
    else:
        url = url_parsed.netloc
    # print(url)
    # print(flask.request.data)
    # flask.request.form['ids'] contains 'stringified' list with obs names, must eval that:
    _ids = ast.literal_eval(flask.request.data)['ids']
    # print(_ids)

    user_id = flask_login.current_user.id
    # get db connection
    client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)

    # trying to get something you're not supposed to get?
    _programs = []
    for _obs in _ids:
        _program, _, _, _, _, _, _ = parse_obs_name(_obs, program_pi)
        _programs.append(_program)

    for _program in set(_programs):
        if user_id != 'admin' and program_pi[_program] != user_id:
            flask.abort(403)

    cursor = coll.find({'_id': {'$in': _ids}, 'distributed.status': True})
    response_text = '#!/usr/bin/env bash\n'
    for obs in cursor:
        _date_str = obs['date_utc'].strftime('%Y%m%d')
        response_text += 'wget http://{:s}/data/{:s}/{:s}/{:s}.tar.bz2\n'.format(url, _date_str,
                                                                                 obs['_id'], obs['_id'])
    # print(response_text)

    # generate .sh file on the fly
    # print(response_text)
    response = flask.make_response(response_text)
    response.headers['Content-Disposition'] = 'attachment; filename=wget.sh'
    return response


# serve root
@app.route('/', methods=['GET'])
@flask_login.login_required
def root():

    if 'start' in flask.request.args:
        start = flask.request.args['start']
    else:
        start = None
    if 'stop' in flask.request.args:
        stop = flask.request.args['stop']
    else:
        stop = None

    user_id = flask_login.current_user.id

    def iter_dates(_dates):
        """
            instead of first loading and then sending everything to user all at once,
             yield data for a single date at a time and stream to user
        :param _dates:
        :return:
        """
        if len(_dates) > 0:
            for _date in sorted(_dates.keys())[::-1]:
                # print(_date, _dates[_date])
                yield _date, _dates[_date]['data'], _dates[_date]['aux']
        else:
            yield None, None, None

    # get db connection
    client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)

    messages = []

    # get science and aux data for all dates:
    dates, msg = get_dates(user_id, coll, coll_aux, start=start, stop=stop, max_dates=30)

    if len(msg) > 0:
        messages += msg

    # provide some feedback:
    if start is not None and stop is not None:
        message = (u'Displaying data from {:s} to {:s}.'.format(start, stop), 'info')
        messages.append(message)
    elif start is None and stop is None:
        message = (u'Displaying last date with observational data.', 'info')
        messages.append(message)
    elif start is None:
        message = (u'Displaying data from now to {:s}.'.format(stop), 'info')
        messages.append(message)
    elif stop is None:
        message = (u'Displaying data from {:s} to now.'.format(start), 'info')
        messages.append(message)

    # note: somehow flashing messages do not work with streaming

    return flask.Response(stream_template('template-archive.html',
                                          user=user_id, start=start, stop=stop,
                                          dates=iter_dates(dates),
                                          current_year=datetime.datetime.now().year,
                                          messages=messages))


# serve root
@app.route('/search', methods=['GET', 'POST'])
@flask_login.login_required
def search():
    """

    :return:
    """
    user_id = flask_login.current_user.id

    # get db connection
    client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)

    # program numbers available for searching:
    # if user_id != 'admin' and _program_pi[_program] != user_id:
    if user_id == 'admin':
        program_ids = program_pi.keys() if len(program_pi.keys()) > 0 else []
    else:
        program_ids = [program_id for program_id in program_pi.keys() if user_id in program_pi[program_id]]

    # got a request?
    if flask.request.method == 'POST':
        # print(flask.request.form)
        # drop indices? no need to do that every time! MongoDB is clever enough to keep things updated
        # coll.drop_indexes()
        # create indices before searching:
        # coll.create_index([('name', 1)])
        # this operation only commences if indices have not been created yet
        try:
            coll.create_index([('coordinates.radec_geojson', '2dsphere'), ('name', 1)])
        except Exception as _e:
            print(_e)

        # query db
        obs, errors = query_db(search_form=flask.request.form, _coll=coll, _program_ids=program_ids, _user_id=user_id)
    else:
        obs = dict()
        errors = []
        # flask.flash('')

    return flask.Response(stream_template('template-search.html', form=flask.request.form,
                                          user=user_id, program_ids=program_ids,
                                          obs=obs, errors=errors,
                                          current_year=datetime.datetime.now().year))


def query_db(search_form, _coll, _program_ids, _user_id):
    """
        parse search form and generate db query
    :param search_form:
    :return:
    """
    # dict to store query to be executed:
    query = dict()
    # list to store the results:
    obs = []
    # list to store errors
    errors = []

    # source name
    source_name = search_form['source_name'].strip()
    # print(source_name)
    source_name_exact = True if ('source_name_exact' in search_form) and search_form['source_name_exact'] == 'on' \
        else False
    if len(source_name) > 0:
        if source_name_exact:
            # exact:
            query['name'] = source_name
            # select = _coll.find({'name': source_name})
        else:
            # contains:
            query['name'] = {'$regex': u'.*{:s}.*'.format(source_name)}
            # select = _coll.find({'name': {'$regex': u'.*{:s}.*'.format(source_name)}})

    # program id:
    program_id = search_form['program_id']
    # security check for non-admin users:
    if _user_id != 'admin' and program_id != 'all':
        assert program_id in _program_ids, \
            'user {:s} tried accessing info that does not belong to him!'.format(_user_id)
    # strict check:
    if program_id == 'all':
        if _user_id != 'admin':
            query['science_program.program_id'] = {'$in': _program_ids}
    else:
        query['science_program.program_id'] = program_id

    # time range:
    try:
        date_range = search_form['daterange'].split('-')
        date_from = date_range[0].strip()
        date_from = datetime.datetime.strptime(date_from, '%Y/%m/%d %H:%M')

        date_to = date_range[1].strip()
        date_to = datetime.datetime.strptime(date_to, '%Y/%m/%d %H:%M')
        query['date_utc'] = {'$gte': date_from, '$lt': date_to}
    except Exception as _e:
        print(_e)
        errors.append('Failed to parse date range')
        return {}, errors

    # position
    ra_str = search_form['ra'].strip()
    dec_str = search_form['dec'].strip()
    if len(ra_str) > 0 and len(dec_str) > 0:
        try:
            # try to guess format and convert to decimal degrees:

            # hms -> ::, dms -> ::
            if ('h' in ra_str) and ('m' in ra_str) and ('s' in ra_str):
                ra_str = ra_str[:-1]  # strip 's' at the end
                for char in ('h', 'm'):
                    ra_str.replace(char, ':')
            if ('d' in dec_str) and ('m' in dec_str) and ('s' in dec_str):
                dec_str = dec_str[:-1]  # strip 's' at the end
                for char in ('d', 'm'):
                    dec_str.replace(char, ':')

            if (':' in ra_str) and (':' in dec_str):
                ra, dec = radec_str2rad(ra_str, dec_str)
            else:
                # already in radian?
                ra = float(ra_str)
                dec = float(dec_str)

            # do cone search
            if 'cone_search_radius' in search_form:
                if len(search_form['cone_search_radius'].strip()) > 0:
                    cone_search_radius = float(search_form['cone_search_radius'])
                    # convert to rad:
                    if search_form['cone_search_unit'] == 'arcsec':
                        cone_search_radius *= np.pi/180.0/3600.
                    elif search_form['cone_search_unit'] == 'arcmin':
                        cone_search_radius *= np.pi/180.0/60.
                    elif search_form['cone_search_unit'] == 'deg':
                        cone_search_radius *= np.pi/180.0

                    # ra, dec in geospatial-friendly format:
                    ra *= 180.0 / np.pi
                    ra -= 180.0
                    dec *= 180.0 / np.pi

                    # print(ra_str, dec_str)
                    # print(ra, dec)
                    # print(cone_search_radius)

                    query['coordinates.radec_geojson'] = {'$geoWithin': {'$centerSphere': [[ra, dec],
                                                                                           cone_search_radius]}}

                else:
                    errors.append('Must specify cone search radius')
                    return {}, errors
            else:
                # (almost) exact match wanted instead?
                pass

        except Exception as _e:
            print(_e)
            errors.append('Failed to recognize RA/Dec format')
            return {}, errors

    # source id
    source_id = search_form['source_id'].strip()
    source_id_exact = True if ('source_id_exact' in search_form) and search_form['source_id_exact'] == 'on' else False
    if len(source_id) > 0:
        if source_id_exact:
            # exact:
            query['_id'] = source_id
        else:
            # contains:
            query['_id'] = {'$regex': u'.*{:s}.*'.format(source_id)}

    # filter:
    filt = search_form['filter']
    if filt != 'any':
        try:
            query['filter'] = get_filter_code(_filter=filt)
        except Exception as e:
            print(e)

    # parse sliders:
    for key in ('seeing_median', 'seeing_nearest',
                'magnitude', 'exposure', 'azimuth', 'elevation', 'lucky_strehl', 'faint_strehl'):
        try:
            val = search_form[key]
            if len(val) > 0:
                rng = list(map(float, val.split(',')))
                # print(key, rng)
                assert rng[0] <= rng[1], 'invalid range for {:s}'.format(key)
                if key == 'azimuth':
                    # not to complicate things, just ignore and don't add to query if full range is requested:
                    if rng == [0.0, 360.0]:
                        continue
                    rng[0] *= np.pi / 180.0
                    rng[1] *= np.pi / 180.0
                    key_query = 'coordinates.azel.0'
                elif key == 'elevation':
                    if rng == [0.0, 90.0]:
                        continue
                    rng[0] *= np.pi / 180.0
                    rng[1] *= np.pi / 180.0
                    key_query = 'coordinates.azel.1'
                elif key == 'lucky_strehl':
                    if rng == [0.0, 100.0]:
                        continue
                    key_query = 'pipelined.bright_star.strehl.ratio_percent'
                elif key == 'faint_strehl':
                    if rng == [0.0, 100.0]:
                        continue
                    key_query = 'pipelined.faint_star.strehl.ratio_percent'
                elif key == 'seeing_nearest':
                    if rng == [0.1, 3.0]:
                        continue
                    key_query = 'seeing.nearest.1'
                elif key == 'seeing_median':
                    if rng == [0.1, 3.0]:
                        continue
                    key_query = 'seeing.median'
                elif key == 'magnitude':
                    if rng == [-6.0, 23.0]:
                        continue
                    key_query = 'magnitude'
                elif key == 'exposure':
                    if rng == [0.0, 600.0]:
                        continue
                    key_query = 'exposure'
                else:
                    key_query = key
                # add to query
                # print(key_query, rng)
                query[key_query] = {'$gte': rng[0], '$lte': rng[1]}
        except Exception as e:
            print(e)
            continue

    # Strehl labeles:
    lucky_strehl_label = search_form['lucky_strehl_label']
    if lucky_strehl_label != 'any':
        try:
            query['pipelined.bright_star.strehl.flag'] = lucky_strehl_label
        except Exception as e:
            print(e)
    faint_strehl_label = search_form['faint_strehl_label']
    if faint_strehl_label != 'any':
        try:
            query['pipelined.faint_star.strehl.flag'] = faint_strehl_label
        except Exception as e:
            print(e)

    # pipelines
    pipe_lucky = True if ('pipe_lucky' in search_form) and search_form['pipe_lucky'] == 'on' else False
    pipe_lucky_pca = True if ('pipe_lucky_pca' in search_form) and search_form['pipe_lucky_pca'] == 'on' else False
    pipe_faint = True if ('pipe_faint' in search_form) and search_form['pipe_faint'] == 'on' else False
    pipe_faint_pca = True if ('pipe_faint_pca' in search_form) and search_form['pipe_faint_pca'] == 'on' else False
    pipe_planetary = True if ('pipe_planetary' in search_form) and search_form['pipe_planetary'] == 'on' else False

    if pipe_lucky:
        query['pipelined.bright_star.status.done'] = True
    if pipe_lucky_pca:
        query['pipelined.bright_star.pca.status.done'] = True
    if pipe_faint:
        query['pipelined.faint_star.status.done'] = True
    if pipe_faint_pca:
        query['pipelined.faint_star.pca.status.done'] = True
    if pipe_planetary:
        query['pipelined.planetary.status.done'] = True

    # execute query:
    if len(query) > 0:
        # print('executing query:\n{:s}'.format(query))

        # don't get/transfer everything -- too much stuff, filter out
        select = _coll.find(query, {'_id': 1, 'seeing': 1, 'science_program': 1, 'name': 1, 'exposure': 1,
                                    'camera': 1, 'magnitude': 1, 'filter': 1, 'date_utc': 1, 'coordinates': 1,
                                    'distributed': 1, 'pipelined.bright_star.status': 1,
                                    'pipelined.bright_star.preview': 1, 'pipelined.bright_star.strehl': 1,
                                    'pipelined.bright_star.pca.status': 1, 'pipelined.bright_star.pca.preview': 1,
                                    'pipelined.faint_star.status': 1,
                                    'pipelined.faint_star.preview': 1, 'pipelined.faint_star.strehl': 1,
                                    'pipelined.faint_star.pca.status': 1, 'pipelined.faint_star.pca.preview': 1
                                    })
        # grab the data
        obs = list(select)

    return obs, errors


# manage users
@app.route('/manage_users')
@flask_login.login_required
def manage_users():
    if flask_login.current_user.id == 'admin':
        # fetch users from the database:
        _users = {}
        client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
        cursor = coll_usr.find()
        for usr in cursor:
            # print(usr)
            if usr['programs'] == 'all':
                _users[usr['_id']] = {'programs': ['all']}
            else:
                _users[usr['_id']] = {'programs': [p.encode('ascii', 'ignore')
                                                   for p in usr['programs']]}

        return flask.render_template('template-users.html',
                                     user=flask_login.current_user.id,
                                     users=_users,
                                     current_year=datetime.datetime.now().year)
    else:
        flask.abort(403)


# manage PSF library
@app.route('/manage_psflib')
@flask_login.login_required
def manage_psflib():
    if flask_login.current_user.id == 'admin':

        if 'start' in flask.request.args:
            start = flask.request.args['start']
        else:
            start = None
        if 'stop' in flask.request.args:
            stop = flask.request.args['stop']
        else:
            stop = None

        user_id = flask_login.current_user.id

        def iter_dates(_dates):
            """
                instead of first loading and then sending everything to user all at once,
                 yield data for a single date at a time and stream to user
            :param _dates:
            :return:
            """
            if len(_dates) > 0:
                for _date in sorted(_dates.keys())[::-1]:
                    # print(_date, _dates[_date])
                    yield _date, _dates[_date]
            else:
                yield None, None

        # get db connection
        client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)

        # get science and aux data for all dates:
        dates = get_dates_psflib(coll_aux, start=start, stop=stop)
        # print(dates)

        return flask.Response(stream_template('template-psflib.html', user=user_id,
                                              start=start, stop=stop,
                                              dates=iter_dates(dates),
                                              num_dates=len(dates),
                                              current_year=datetime.datetime.now().year))

    else:
        flask.abort(403)


@app.route('/add_user', methods=['GET'])
@flask_login.login_required
def add_user():
    try:
        user = flask.request.args['user']
        password = flask.request.args['password']
        programs = [p.strip().encode('ascii', 'ignore')
                    for p in flask.request.args['programs'].split(',')]
        # print(user, password, programs)
        # print(len(user), len(password), len(programs))
        if len(user) == 0 or len(password) == 0:
            return 'everything must be set'
        client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
        # add user to coll_usr collection:
        coll_usr.insert_one(
            {'_id': user,
             'password': generate_password_hash(password),
             'programs': programs,
             'last_modified': datetime.datetime.now()}
        )
        # change data ownership (append to program_PI) in the main database.
        # if program_PI == ['admin'], pull admin out of it
        if user != 'admin':  # admin has it 'all'!
            coll.update({'science_program.program_id': {'$in': programs}},
                        {'$push': {'science_program.program_PI': user}}, multi=True)
            # pull 'admin' out of list if program existed, but did not have an owner
            coll.update({'science_program.program_id': {'$in': programs}},
                        {'$pull': {'science_program.program_PI': 'admin'}}, multi=True)

        return 'success'
    except Exception as _e:
        print(_e)
        return _e


@app.route('/edit_user', methods=['GET'])
@flask_login.login_required
def edit_user():
    try:
        # print(flask.request.args)
        id = flask.request.args['_user']
        if id == 'admin':
            return 'Cannot remove the admin!'
        user = flask.request.args['edit-user']
        password = flask.request.args['edit-password']
        programs = [p.strip().encode('ascii', 'ignore')
                    for p in flask.request.args['edit-programs'].split(',')]
        # print(user, password, programs, id)
        # print(len(user), len(password), len(programs))
        if len(user) == 0:
            return 'username must be set'
        # keep old password:
        if len(password) == 0:
            client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
            result = coll_usr.update_one(
                {'_id': id},
                {
                    '$set': {
                        '_id': user,
                        'programs': programs
                    },
                    '$currentDate': {'last_modified': True}
                }
            )
        # else change password too:
        else:
            client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
            result = coll_usr.update_one(
                {'_id': id},
                {
                    '$set': {
                        '_id': user,
                        'password': generate_password_hash(password),
                        'programs': programs
                    },
                    '$currentDate': {'last_modified': True}
                }
            )

        # change program ownership if necessary:
        # get all program numbers for this user:
        programs_in_db = [k for k in program_pi.keys() if user in program_pi[k]]
        old_set = set(programs_in_db)
        new_set = set(programs)
        # change data ownership in the main database:
        if user != 'admin':  # admin has it 'all'!
            # added new programs?
            if len(new_set - old_set) > 0:
                # reset ownership for the set difference new-old (program in new, but not in old)
                coll.update({'science_program.program_id': {'$in': list(new_set - old_set)}},
                            {'$push': {'science_program.program_PI': user}}, multi=True)
                # pull 'admin' out of list if program did not have an owner
                coll.update({'science_program.program_id': {'$in': list(new_set - old_set)}},
                            {'$pull': {'science_program.program_PI': 'admin'}}, multi=True)
            # removed programs? reset ownership
            if len(old_set - new_set) > 0:
                # reset ownership for the set difference old-new (program in old, but not in new)
                coll.update({'science_program.program_id': {'$in': list(old_set - new_set)}},
                            {'$pull': {'science_program.program_PI': user}}, multi=True)
                # change to ['admin'] if ended up being empty
                coll.update({'science_program.program_id': {'$in': list(old_set - new_set)},
                             'science_program.program_PI': {'$eq': []}},
                            {'$push': {'science_program.program_PI': 'admin'}}, multi=True)
        return 'success'
    except Exception as _e:
        print(_e)
        return _e


@app.route('/remove_user', methods=['GET', 'POST'])
@flask_login.login_required
def remove_user():
    try:
        # print(flask.request.args)
        # get username from request
        user = flask.request.args['user']
        if user == 'admin':
            return 'Cannot remove the admin!'
        # print(user)

        # get db:
        client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
        # get all program numbers for this user:
        user_programs = [k for k in program_pi.keys() if user in program_pi[k]]
        # try to remove the user:
        coll_usr.delete_one({'_id': user})
        # change data ownership to 'admin':
        # pull user first
        coll.update({'science_program.program_id': {'$in': user_programs}},
                    {'$pull': {'science_program.program_PI': user}}, multi=True)
        # change to ['admin'] if empty
        coll.update({'science_program.program_id': {'$in': user_programs},
                     'science_program.program_PI': {'$eq': []}},
                    {'$push': {'science_program.program_PI': 'admin'}}, multi=True)

        return 'success'
    except Exception as _e:
        print(_e)
        return _e


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    flask_login.logout_user()
    return flask.redirect(flask.url_for('root'))


@app.errorhandler(500)
def internal_error(error):
    return '500 error'


@app.errorhandler(404)
def not_found(error):
    return '404 error'


@app.errorhandler(403)
def not_found(error):
    return '403 error: forbidden'


@login_manager.unauthorized_handler
def unauthorized_handler():
    return flask.redirect(flask.url_for('login'))


if __name__ == '__main__':
    app.run(host=config['server']['host'], port=config['server']['port'], threaded=True)
