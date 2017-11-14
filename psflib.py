import matplotlib
# matplotlib.use('Qt5Agg')
from pymongo.errors import ExecutionTimeout

matplotlib.use('Agg')

import os
from pymongo import MongoClient, DESCENDING
import inspect
import json
import datetime
import pytz
import time
import argparse
import logging
import sys
import numpy as np
from astropy.io import fits
import traceback
import ast
import re
import vip

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, TextArea
from matplotlib.patches import Rectangle
# import matplotlib.mlab as mlab
# import seaborn as sns
#
# # set up plotting
# sns.set_style('whitegrid')
# plt.close('all')
# sns.set_context('talk')


# Scale bars
class AnchoredSizeBar(AnchoredOffsetbox):
    def __init__(self, transform, size, label, loc,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, frameon=True):
        """
        Draw a horizontal bar with the size in data coordinate of the give axes.
        A label will be drawn underneath (center-aligned).

        pad, borderpad in fraction of the legend font size (or prop)
        sep in points.
        loc:
            'upper right'  : 1,
            'upper left'   : 2,
            'lower left'   : 3,
            'lower right'  : 4,
            'right'        : 5,
            'center left'  : 6,
            'center right' : 7,
            'lower center' : 8,
            'upper center' : 9,
            'center'       : 10
        """
        self.size_bar = AuxTransformBox(transform)
        self.size_bar.add_artist(Rectangle((0, 0), size, 0, fc='none', color='white', lw=3))

        self.txt_label = TextArea(label, dict(color='white', size='x-large', weight='normal'),
                                  minimumdescent=False)

        self._box = VPacker(children=[self.size_bar, self.txt_label],
                            align="center",
                            pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box,
                                   prop=prop,
                                   frameon=frameon)


def get_xy_from_pipeline_settings_txt(_path, _first=True):
    """
        Get centroid position for a lucky-pipelined image
    :param _path:
    :param _first: output the x,y from the first run? if False, output from the last
    :return:
    """
    with open(os.path.join(_path, 'pipeline_settings.txt'), 'r') as _f:
        f_lines = _f.readlines()

    for l in f_lines:
        _tmp = re.search(r'\d\s+\d\s+\((\d+),(\d+)\),\((\d+),(\d+)\)\n', l)
        if _tmp is not None:
            _x = (int(_tmp.group(1)) + int(_tmp.group(3)))/2
            _y = (int(_tmp.group(2)) + int(_tmp.group(4)))/2
            if _first:
                break

    return _x, _y


def utc_now():
    return datetime.datetime.now(pytz.utc)


def get_config(_config_file='config.json'):
    """
        Load config JSON file
    """
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


def connect_to_db(_config, _logger=None):
    """ Connect to the mongodb database

    :return:
    """
    try:
        if _logger is not None:
            _logger.debug('Connecting to the Robo-AO database at {:s}:{:d}'.
                              format(_config['database']['host'], _config['database']['port']))
        if _config['server']['environment'] == 'production':
            # in production, must set up replica set
            _client = MongoClient(host=_config['database']['host'], port=_config['database']['port'],
                                  replicaset=_config['database']['replicaset'],
                                  readPreference='primaryPreferred')
        else:
            # standalone from my laptop, when there's only one instance of DB
            _client = MongoClient(host=_config['database']['host'], port=_config['database']['port'])
        # grab main database:
        _db = _client[_config['database']['db']]
    except Exception as _e:
        if _logger is not None:
            _logger.error(_e)
            _logger.error('Failed to connect to the Robo-AO database at {:s}:{:d}'.
                          format(_config['database']['host'], _config['database']['port']))
        # raise error
        raise ConnectionRefusedError
    try:
        # authenticate
        _db.authenticate(_config['database']['user'], _config['database']['pwd'])
        if _logger is not None:
            _logger.debug('Successfully authenticated with the Robo-AO database at {:s}:{:d}'.
                          format(_config['database']['host'], _config['database']['port']))
    except Exception as _e:
        if _logger is not None:
            _logger.error(_e)
            _logger.error('Authentication failed for the Robo-AO database at {:s}:{:d}'.
                          format(_config['database']['host'], _config['database']['port']))
        raise ConnectionRefusedError
    try:
        # get collection with observations
        _coll_obs = _db[_config['database']['collection_obs']]
        if _logger is not None:
            _logger.debug('Using collection {:s} with obs data in the database'.
                          format(_config['database']['collection_obs']))
    except Exception as _e:
        if _logger is not None:
            _logger.error(_e)
            _logger.error('Failed to use a collection {:s} with obs data in the database'.
                          format(_config['database']['collection_obs']))
        raise NameError
    try:
        # get collection with auxiliary stuff
        _coll_aux = _db[_config['database']['collection_aux']]
        if _logger is not None:
            _logger.debug('Using collection {:s} with aux data in the database'.
                          format(_config['database']['collection_aux']))
    except Exception as _e:
        if _logger is not None:
            _logger.error(_e)
            _logger.error('Failed to use a collection {:s} with aux data in the database'.
                          format(_config['database']['collection_aux']))
        raise NameError
    _coll_weather = None
    try:
        # get collection with user access credentials
        _coll_usr = _db[_config['database']['collection_pwd']]
        if _logger is not None:
            _logger.debug('Using collection {:s} with user access credentials in the database'.
                          format(_config['database']['collection_pwd']))
    except Exception as _e:
        if _logger is not None:
            _logger.error(_e)
            _logger.error('Failed to use a collection {:s} with user access credentials in the database'.
                          format(_config['database']['collection_pwd']))
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
        if _logger is not None:
            _logger.error(_e)

    if _logger is not None:
        _logger.debug('Successfully connected to Robo-AO database at {:s}:{:d}'.
                      format(_config['database']['host'], _config['database']['port']))

    return _client, _db, _coll_obs, _coll_aux, _coll_weather, _program_pi


def remove_from_lib(_psf_library_fits, _obs):
    try:
        with fits.open(_psf_library_fits) as hdulist:
            # print(hdulist[0].data.shape)
            if hdulist[0].data.shape[0] < 3:
                raise Exception('Kennot remove {:s}! Must have at least five PSF in the library'.format(_obs))

            # get index of the frame to be removed:
            index_obs = np.argmax(hdulist[-1].data['obs_names'] == _obs)

            # remove from table with names:
            hdulist[-1].data = np.delete(hdulist[-1].data, index_obs, axis=0)
            # remove from images:
            hdulist[0].data = np.delete(hdulist[0].data, index_obs, axis=0)
            # update library:
            hdulist.writeto(_psf_library_fits, overwrite=True)
    except Exception as _e:
        traceback.print_exc()
        print(_e)
        raise Exception('Failed to remove {:s} from PSF library'.format(_obs))


def add_to_lib(_psf_library_fits, _path, _obs, _obj_name='unknown'):
    # first extension HDU contains a table with obs names and short names, the first contains actual PSFs
    try:
        # get frame:
        with fits.open(_path) as f:
            frame = np.nan_to_num(f[0].data)

        if not os.path.exists(_psf_library_fits):
            # library does not exist yet?
            frame_names = np.array([_obs])
            frame_short_names = np.array([_obj_name])
            # create a binary table to keep full obs names and "short" (object) names
            tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='obs_names', format='80A',
                                                               array=frame_names),
                                                   fits.Column(name='obj_names', format='80A',
                                                               array=frame_short_names)])
            thdulist = fits.HDUList([fits.PrimaryHDU([frame]), tbhdu])
            thdulist.writeto(_psf_library_fits, overwrite=True)
        else:
            # get library:
            with fits.open(_psf_library_fits) as hdulist:
                # append names:
                tbhdu = fits.BinTableHDU.from_columns(hdulist[-1].columns, nrows=hdulist[-1].data.shape[0]+1)
                tbhdu.data['obs_names'][-1] = _obs
                tbhdu.data['obj_names'][-1] = _obj_name
                hdulist[-1] = tbhdu
                # append frame:
                hdulist[0].data = np.vstack((hdulist[0].data, [frame]))
                # update library:
                hdulist.writeto(_psf_library_fits, overwrite=True)

    except:
        traceback.print_exc()
        raise Exception('Failed to add {:s} to PSF library'.format(_obs))


def in_fits(_psf_library_fits, _obs):
    """
        Check if _obs is in PSF library
    :param _psf_library_fits:
    :param _obs:
    :return:
    """
    try:
        with fits.open(_psf_library_fits) as hdulist:
            if _obs in hdulist[-1].data['obs_names']:
                return True
            else:
                return False
    except Exception as _e:
        traceback.print_exc()
        print(_e)
        return False


def set_up_logging(_path='logs', _name='archive', _level=logging.DEBUG, _mode='w'):
    """ Set up logging

    :param _path:
    :param _name:
    :param _level: DEBUG, INFO, etc.
    :param _mode: overwrite log-file or append: w or a
    :return: logger instance
    """

    if not os.path.exists(_path):
        os.makedirs(_path)
    utc_now = datetime.datetime.utcnow()

    # http://www.blog.pythonlibrary.org/2012/08/02/python-101-an-intro-to-logging/
    _logger = logging.getLogger(_name)

    _logger.setLevel(_level)
    # create the logging file handler
    fh = logging.FileHandler(os.path.join(_path,
                                          '{:s}.{:s}.log'.format(_name, utc_now.strftime('%Y%m%d'))),
                             mode=_mode)
    logging.Formatter.converter = time.gmtime

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)

    # add handler to logger object
    _logger.addHandler(fh)

    return _logger, utc_now.strftime('%Y%m%d')


def shut_down_logger(_logger):
    """
        prevent writing to multiple log-files after 'manual rollover'
    :param _logger:
    :return:
    """
    handlers = _logger.handlers[:]
    for handler in handlers:
        handler.close()
        _logger.removeHandler(handler)


def naptime(_config):
    """
        Return time to sleep in seconds for the archiving engine
        before waking up to rerun itself.
         In the daytime, it's 1 hour
         In the nap time, it's nap_time_start_utc - utc_now()
    :return:
    """
    try:
        # local or UTC?
        tz = pytz.utc if _config['misc']['nap_time_frame'] == 'UTC' else None
        now = datetime.datetime.now(tz)

        if _config['misc']['nap_at_night']:

            last_midnight = datetime.datetime(now.year, now.month, now.day, tzinfo=tz)
            next_midnight = datetime.datetime(now.year, now.month, now.day, tzinfo=tz) \
                            + datetime.timedelta(days=1)

            hm_start = list(map(int, _config['misc']['nap_time_start'].split(':')))
            hm_stop = list(map(int, _config['misc']['nap_time_stop'].split(':')))

            if hm_stop[0] < hm_start[0]:
                h_before_midnight = 24 - (hm_start[0] + hm_start[1] / 60.0)
                h_after_midnight = hm_stop[0] + hm_stop[1] / 60.0

                # print((next_midnight - now).total_seconds() / 3600.0, h_before_midnight)
                # print((now - last_midnight).total_seconds() / 3600.0, h_after_midnight)

                if (next_midnight - now).total_seconds() / 3600.0 < h_before_midnight:
                    sleep_until = next_midnight + datetime.timedelta(hours=h_after_midnight)
                    print('sleep until:', sleep_until)
                elif (now - last_midnight).total_seconds() / 3600.0 < h_after_midnight:
                    sleep_until = last_midnight + datetime.timedelta(hours=h_after_midnight)
                    print('sleep until:', sleep_until)
                else:
                    sleep_until = now + datetime.timedelta(minutes=_config['misc']['loop_interval'])
                    print('sleep until:', sleep_until)

            else:
                h_after_midnight_start = hm_start[0] + hm_start[1] / 60.0
                h_after_midnight_stop = hm_stop[0] + hm_stop[1] / 60.0

                if (last_midnight + datetime.timedelta(hours=h_after_midnight_start) <
                        now < last_midnight + datetime.timedelta(hours=h_after_midnight_stop)):
                    sleep_until = last_midnight + datetime.timedelta(hours=h_after_midnight_stop)
                    print('sleep until:', sleep_until)
                else:
                    sleep_until = now + datetime.timedelta(minutes=_config['misc']['loop_interval'])
                print('sleep until:', sleep_until)

            return (sleep_until - now).total_seconds()

        else:
            # sleep for loop_interval minutes otherwise (return seconds)
            return _config['misc']['loop_interval'] * 60.0

    except Exception as _e:
        print(_e)
        traceback.print_exc()
        # return _config['misc']['loop_interval']*60
        # sys.exit()
        return False


if __name__ == '__main__':
    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Manage Robo-AO\'s PSF library')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)

    args = parser.parse_args()

    ''' set up logging at init '''
    logger, logger_utc_date = set_up_logging(_path='logs', _name='psflib', _level=logging.INFO, _mode='a')

    while True:
        # check if a new log file needs to be started:
        if datetime.datetime.utcnow().strftime('%Y%m%d') != logger_utc_date:
            # reset
            shut_down_logger(logger)
            logger, logger_utc_date = set_up_logging(_path='logs', _name='psflib', _level=logging.INFO, _mode='a')

        # if you start me up... if you start me up I'll never stop (hopefully not)
        logger.info('Started cycle.')

        try:
            ''' load config data '''
            try:
                config = get_config(_config_file=args.config_file)
                logger.debug('Successfully read in the config file {:s}'.format(args.config_file))
            except Exception as e:
                traceback.print_exc()
                logger.error(e)
                logger.error('Failed to read in the config file {:s}'.format(args.config_file))
                sys.exit()

            ''' Connect to the mongodb database '''
            try:
                client, db, coll, coll_aux, coll_weather, program_pi = connect_to_db(_config=config, _logger=logger)
                if None in (db, coll, program_pi):
                    raise Exception('Failed to connect to the database')
            except Exception as e:
                traceback.print_exc()
                logger.error(e)
                sys.exit()

            '''
             #############################
             COMMENCE OPERATION PROCESSING
             #############################
            '''
            # PSF library fits file name
            # psf_library_fits = os.path.join(config['path_archive'], 'psf_library.fits')
            psf_library_fits = config['pipeline']['bright_star']['high_contrast']['path_psf_reference_library']

            ''' get aux data '''
            print('Archiving start date: ', config['misc']['archiving_start_date'])
            # select_aux = coll_aux.find({'_id': {'$gte': config['archiving_start_date'].strftime('%Y%m%d')}},
            #                            max_time_ms=20000)
            select_aux = coll_aux.find({'_id': {'$gte': config['misc']['archiving_start_date']}})

            if select_aux.count() > 0:

                # iterate over all dates from coll_aux, from recent to older:
                for ob_aux in select_aux.sort([('_id', DESCENDING)]):

                    date_str = ob_aux['_id']
                    date = datetime.datetime.strptime(ob_aux['_id'], '%Y%m%d')

                    print(date)

                    # path to store data for individual frames:
                    _path_out = os.path.join(config['path']['path_archive'], date_str, 'summary', 'psflib')

                    ''' TODO: create psflib field in aux collection in the database, or make sure it's there '''
                    # check when last updated
                    if 'psf_lib' not in ob_aux:
                        # init if necessary
                        coll_aux.update_one(
                            {'_id': date_str},
                            {
                                '$set': {
                                    'psf_lib': {}
                                }
                            }
                        )
                        # reload:
                        ob_aux = coll_aux.find_one({'_id': date_str}, max_time_ms=10000)
                        # frames: {'obs_name': {'in_library': False, 'outdated': False,
                        #                       'failed': False, 'last_modified': datetime}

                    ''' grab obs credentials from the database '''
                    # dict to store query to be executed on the main collection (with obs data):
                    query = dict()

                    # query['filter'] = 'Si'

                    # query['seeing.median'] = {'$ne': None}

                    query['date_utc'] = {'$gte': date, '$lt': date + datetime.timedelta(days=1)}

                    # exclude planetary data:
                    query['science_program.program_id'] = {'$ne': 24}

                    # azimuth and elevation range:
                    # query['coordinates.azel.0'] = {'$gte': 0 * np.pi / 180.0, '$lte': 360 * np.pi / 180.0}
                    # query['coordinates.azel.1'] = {'$gte': 0 * np.pi / 180.0, '$lte': 90 * np.pi / 180.0}

                    # consider reliable Strehls only:
                    # query['pipelined.bright_star.strehl.flag'] = {'$eq': 'OK'}

                    # consider only 'done' stuff:
                    query['pipelined.bright_star.status.done'] = True

                    # discard observations marked as "zero_flux" and "failed" by the bright_star pipeline
                    query['pipelined.bright_star.classified_as'] = {'$nin': ['zero_flux', 'failed']}

                    # execute query:
                    if len(query) > 0:
                        # print('executing query:\n{:s}'.format(query))
                        # select = coll.find(query, max_time_ms=20000)
                        select = coll.find(query)

                        if select.count() > 0:

                            for ob in select:

                                try:
                                    # see lucidchart.com for the processing flowchart

                                    ob_id = ob['_id']
                                    # field names (keys) in MongoDB cannot contain dots, so let's replace them with _:
                                    ob_id_db = ob['_id'].replace('.', '_')

                                    in_db = ob_id_db in ob_aux['psf_lib']

                                    # last pipelined:
                                    last_pipelined = ob['pipelined']['bright_star']['last_modified']

                                    execute_processing = False

                                    ''' in DB? '''
                                    if in_db:
                                        ''' Done? '''
                                        done = ob_aux['psf_lib'][ob_id_db]['done']
                                        if done:
                                            ''' Updated? '''
                                            updated = np.abs(last_pipelined -
                                                    ob_aux['psf_lib'][ob_id_db]['last_modified']).total_seconds() > 60
                                            if updated:
                                                # mark in DB:
                                                coll_aux.update_one(
                                                    {'_id': date_str},
                                                    {
                                                        '$set': {
                                                            'psf_lib.{:s}.done'.format(ob_id_db): False,
                                                            'psf_lib.{:s}.updated'.format(ob_id_db): True,
                                                            'psf_lib.{:s}.last_modified'.format(ob_id_db):
                                                                last_pipelined
                                                        }
                                                    }
                                                )
                                                #
                                                execute_processing = True
                                                logger.info('{:s} was modified, updated entry'.format(ob_id))

                                            ''' Check status from web interface '''
                                            enqueued = ob_aux['psf_lib'][ob_id_db]['enqueued']
                                            if enqueued:
                                                status = ob_aux['psf_lib'][ob_id_db]['status']
                                                if status == 'add_to_lib':
                                                    try:
                                                        in_lib = ob_aux['psf_lib'][ob_id_db]['in_lib']
                                                        if in_lib or updated:
                                                            # remove from lib, mark updated: false
                                                            remove_from_lib(_psf_library_fits=psf_library_fits,
                                                                            _obs=ob_id_db)
                                                            # execute add to psf lib
                                                            add_to_lib(_psf_library_fits=psf_library_fits,
                                                                       _path=os.path.join(_path_out,
                                                                                          '{:s}.fits'.format(ob_id_db)),
                                                                       _obs=ob_id_db, _obj_name=ob['name'])
                                                        else:
                                                            # execute add to psf lib
                                                            add_to_lib(_psf_library_fits=psf_library_fits,
                                                                       _path=os.path.join(_path_out,
                                                                                          '{:s}.fits'.format(ob_id_db)),
                                                                       _obs=ob_id_db, _obj_name=ob['name'])

                                                        coll_aux.update_one(
                                                            {'_id': date_str},
                                                            {
                                                                '$set': {
                                                                    'psf_lib.{:s}.in_lib'.format(ob_id_db): True
                                                                }
                                                            }
                                                        )
                                                        logger.info('{:s} added to the library'.format(ob_id))
                                                    except Exception as e:
                                                        print(e)
                                                        coll_aux.update_one(
                                                            {'_id': date_str},
                                                            {
                                                                '$set': {
                                                                    'psf_lib.{:s}.in_lib'.format(ob_id_db): False
                                                                }
                                                            }
                                                        )
                                                        logger.error('failed to add {:s} to the library'.format(ob_id))

                                                elif status == 'remove_from_lib':
                                                    try:
                                                        # check if actually in lib
                                                        if in_fits(_psf_library_fits=psf_library_fits, _obs=ob_id_db):
                                                            # remove from lib
                                                            remove_from_lib(_psf_library_fits=psf_library_fits,
                                                                            _obs=ob_id_db)
                                                            logger.info('removed {:s} from the library'.format(ob_id))
                                                        else:
                                                            logger.info('{:s} not in the library, cannot remove'.
                                                                        format(ob_id))
                                                    except Exception as e:
                                                        print(e)
                                                        logger.error('failed to remove {:s}'.format(ob_id))

                                                    coll_aux.update_one(
                                                        {'_id': date_str},
                                                        {
                                                            '$set': {
                                                                'psf_lib.{:s}.in_lib'.format(ob_id_db): False
                                                            }
                                                        }
                                                    )

                                                coll_aux.update_one(
                                                    {'_id': date_str},
                                                    {
                                                        '$set': {
                                                            'psf_lib.{:s}.updated'.format(ob_id_db): False,
                                                            'psf_lib.{:s}.enqueued'.format(ob_id_db): False,
                                                            'psf_lib.{:s}.status'.format(ob_id_db): None
                                                        }
                                                    }
                                                )
                                            else:
                                                # not enqueued? check if marked correctly:
                                                in_lib = ob_aux['psf_lib'][ob_id_db]['in_lib']

                                                if in_lib != in_fits(_psf_library_fits=psf_library_fits, _obs=ob_id_db):
                                                    coll_aux.update_one(
                                                        {'_id': date_str},
                                                        {
                                                            '$set': {
                                                                'psf_lib.{:s}.in_lib'.format(ob_id_db):
                                                                    in_fits(_psf_library_fits=psf_library_fits,
                                                                            _obs=ob_id_db)
                                                            }
                                                        }
                                                    )
                                                    logger.info('corrected {:s} status in database'.format(ob_id))

                                        else:
                                            ''' Process if tried not too many times and high_flux or faint '''
                                            tried_too_many_times = ob_aux['psf_lib'][ob_id_db]['retries'] \
                                                                    > config['misc']['max_retries']
                                            tag = str(ob['pipelined']['bright_star']['classified_as'])
                                            if (not tried_too_many_times) and (tag not in ('failed', 'zero_flux')):
                                                execute_processing = True

                                    else:
                                        # init entry for a new record:
                                        coll_aux.update_one(
                                            {'_id': date_str},
                                            {
                                                '$set': {
                                                    'psf_lib.{:s}'.format(ob_id_db): {
                                                        'done': False,
                                                        'in_lib': False,
                                                        'updated': False,
                                                        'enqueued': False,
                                                        'status': False,
                                                        'retries': 0,
                                                        'last_modified': utc_now()
                                                    }
                                                }
                                            }
                                        )
                                        #
                                        execute_processing = True
                                        logger.info('initialized {:s}'.format(ob_id))

                                    if execute_processing:

                                        # number of processing attempts++
                                        coll_aux.update_one(
                                            {'_id': date_str},
                                            {
                                                '$inc': {
                                                        'psf_lib.{:s}.retries'.format(ob_id_db): 1
                                                    }
                                            }
                                        )

                                        ''' preprocess 100p.fits, high-pass, cut '''
                                        _path = os.path.join(config['path']['path_archive'], date_str,
                                                             ob_id, 'bright_star')
                                        _fits_name = '100p.fits'

                                        if os.path.exists(os.path.join(_path, _fits_name)):

                                            print(os.path.join(_path, _fits_name))

                                            with fits.open(os.path.join(_path, _fits_name)) as _hdu:
                                                scidata = np.nan_to_num(_hdu[0].data)

                                            _win = config['pipeline']['bright_star']['high_contrast']['win']
                                            y, x = get_xy_from_pipeline_settings_txt(_path)
                                            # drizzled
                                            x *= 2.0
                                            y *= 2.0
                                            x, y = list(map(int, [x, y]))

                                            # out of the frame? do not try to fix that, just skip!
                                            if x - _win < 0 or x + _win + 1 >= scidata.shape[0] \
                                                    or y - _win < 0 or y + _win + 1 >= scidata.shape[1]:
                                                coll_aux.update_one(
                                                    {'_id': date_str},
                                                    {
                                                        '$set': {
                                                            'psf_lib.{:s}.done'.format(ob_id_db): False,
                                                            'psf_lib.{:s}.last_modified'.format(ob_id_db): utc_now()
                                                        }
                                                    }
                                                )
                                                continue

                                            _trimmed_frame = scidata[x - _win: x + _win + 1,
                                                                     y - _win: y + _win + 1]

                                            # Filter the trimmed frame with IUWT filter, 2 coeffs
                                            filtered_frame = (vip.var.cube_filter_iuwt(
                                                np.reshape(_trimmed_frame, (1, np.shape(_trimmed_frame)[0],
                                                            np.shape(_trimmed_frame)[1])),
                                                coeff=5, rel_coeff=2))
                                            print(filtered_frame.shape)

                                            _fit = vip.var.fit_2dgaussian(filtered_frame[0], crop=True, cropsize=50,
                                                                          debug=False, full_output=True)
                                            fwhm_y = float(_fit['fwhm_y'])
                                            fwhm_x = float(_fit['fwhm_x'])
                                            # amplitude = float(_fit['amplitude'])
                                            # theta = float(_fit['theta'])

                                            _fwhm = np.mean([fwhm_y, fwhm_x])

                                            # Print the resolution element size
                                            # print('Using resolution element size = ', _fwhm)
                                            if _fwhm < 2:
                                                _fwhm = 2.0
                                                # print('Too small, changing to ', _fwhm)
                                            _fwhm = int(_fwhm)

                                            # Center the filtered frame
                                            centered_cube, shy, shx = \
                                                (vip.preproc.cube_recenter_gauss2d_fit(array=filtered_frame,
                                                                                       xy=(_win, _win), fwhm=_fwhm,
                                                                                       subi_size=6, nproc=1,
                                                                                       full_output=True))

                                            centered_frame = centered_cube[0]

                                            ''' create preview '''
                                            fig = plt.figure()
                                            ax = fig.add_subplot(111)

                                            ''' cropped image: '''
                                            # save cropped image
                                            plt.close('all')
                                            fig = plt.figure()
                                            fig.set_size_inches(3, 3, forward=False)
                                            # ax = fig.add_subplot(111)
                                            ax = plt.Axes(fig, [0., 0., 1., 1.])
                                            ax.set_axis_off()
                                            fig.add_axes(ax)
                                            ax.imshow(centered_frame, cmap=plt.cm.magma, origin='lower',
                                                      interpolation='nearest')

                                            # add scale bar:
                                            _drizzled = True
                                            _fow_x = 36
                                            _pix_x = 1024
                                            # draw a horizontal bar with length of 0.1*x_size
                                            # (ax.transData) with a label underneath.
                                            bar_len = centered_frame.shape[0] * 0.1
                                            mltplr = 2 if _drizzled else 1
                                            bar_len_str = '{:.1f}'.format(bar_len * _fow_x / _pix_x / mltplr)
                                            asb = AnchoredSizeBar(ax.transData,
                                                                  bar_len,
                                                                  bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" +
                                                                  bar_len_str[-1],
                                                                  loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
                                            ax.add_artist(asb)

                                            # plt.show()

                                            ''' store both fits and png for the web interface '''
                                            png_name = '{:s}.png'.format(ob_id_db)
                                            fits_name = '{:s}.png'.format(ob_id_db)
                                            if not (os.path.exists(_path_out)):
                                                os.makedirs(_path_out)
                                            fig.savefig(os.path.join(_path_out, png_name), dpi=300)
                                            # save box around selected object:
                                            hdu = fits.PrimaryHDU(centered_frame)
                                            hdu.writeto(os.path.join(_path_out, '{:s}.fits'.format(ob_id_db)),
                                                        overwrite=True)

                                            # mark done:
                                            coll_aux.update_one(
                                                {'_id': date_str},
                                                {
                                                    '$set': {
                                                        'psf_lib.{:s}.done'.format(ob_id_db): True,
                                                        'psf_lib.{:s}.last_modified'.format(ob_id_db): last_pipelined
                                                    }
                                                }
                                            )
                                            logger.info('successfully processed {:s}'.format(ob_id))

                                except Exception as e:
                                    traceback.print_exc()
                                    print(e)
                                    coll_aux.update_one(
                                        {'_id': date_str},
                                        {
                                            '$set': {
                                                'psf_lib.{:s}.done'.format(ob_id_db): False,
                                                'psf_lib.{:s}.last_modified'.format(ob_id_db): utc_now()
                                            }
                                        }
                                    )
                                    logger.error('failed to process {:s}'.format(ob_id))

                logger.info('Finished cycle.')
                sleep_for = naptime(config)  # seconds
                if sleep_for:
                    # disconnect from database not to keep the connection alive while sleeping
                    if 'client' in locals():
                        client.close()
                    time.sleep(sleep_for)
                else:
                    logger.error('Could not fall asleep, exiting.')
                    break

        except ExecutionTimeout:
            print('query took too long to execute, falling asleep, will retry later')
            sleep_for = naptime(config)  # seconds
            if sleep_for:
                # disconnect from database not to keep the connection alive while sleeping
                if 'client' in locals():
                    client.close()
                time.sleep(sleep_for)
                continue
            else:
                logger.error('Could not fall asleep, exiting.')
                break

        except KeyboardInterrupt:
            logger.error('User exited.')
            logger.info('Finished cycle.')
            # try disconnecting from the database (if connected)
            try:
                if 'client' in locals():
                    client.close()
            finally:
                break

        except Exception as e:
            print(e)
            traceback.print_exc()
            logger.error(e)
            logger.error('Unknown error, exiting. Please check the logs')
            # TODO: send out an email with an alert
            logger.info('Finished cycle.')
            # try disconnecting from the database (if connected)
            try:
                if 'client' in locals():
                    client.close()
            finally:
                break
