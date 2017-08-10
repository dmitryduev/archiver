import argparse
import signal
import pytz
from distributed import Client, LocalCluster
from queue import Queue
import time
import datetime
import inspect
import os
import subprocess
import numpy as np
import glob
import traceback
import sys
import json
import logging
import collections
import pymongo
import re
from astropy.io import fits
import pyprind


class TimeoutError(Exception):
    def __init__(self, value="operation timed out"):
        self.value = value

    def __str__(self):
        return repr(self.value)


def timeout(seconds_before_timeout):
    """
        A decorator that raises a TimeoutError error if a function/method runs longer than seconds_before_timeout
    :param seconds_before_timeout:
    :return:
    """
    def decorate(f):
        def handler(signum, frame):
            raise TimeoutError()

        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            old_time_left = signal.alarm(seconds_before_timeout)
            if 0 < old_time_left < seconds_before_timeout:  # never lengthen existing timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
            finally:
                if old_time_left > 0:  # deduct f's run time from the saved timer
                    old_time_left -= time.time() - start_time
                signal.signal(signal.SIGALRM, old)
                signal.alarm(old_time_left)
            return result
        new_f.func_name = f.func_name
        return new_f
    return decorate


class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


class SetQueue(Queue):
    def _init(self, maxsize):
        self.queue = set()

    def _put(self, item):

        if item in self.queue:
            # do not count it since it's not going to be added to the set!
            # see line 144 in queue.py (python 3.6)
            self.unfinished_tasks -= 1
        self.queue.add(item)

    def _get(self):
        return self.queue.pop()


class OrderedSetQueue(SetQueue):
    def _init(self, maxsize):
        self.queue = OrderedSet()


class Archiver(object):
    """
        A class representation of major data house-keeping/archiving tasks
    """
    def __init__(self, config_file=None):
        assert config_file is not None, 'Must specify config file'

        try:
            ''' load config data '''
            self.config = self.get_config(_config_file=config_file)

            ''' set up logging at init '''
            self.logger, self.logger_utc_date = self.set_up_logging(_name='archive', _mode='a')

            ''' initialize dask.distributed LocalCluster for distributed task processing '''
            # alternative, or if workers are to be run on different machines
            '''
            In different terminals start the scheduler and a few workers:
            $ dask-scheduler
            $ dask-worker 127.0.0.1:8786 --nprocs 2 --nthreads 1
            $ dask-worker 127.0.0.1:8786 --nprocs 2 --nthreads 1
            $ ...
            then here:
            self.c = Client('127.0.0.1:8786')
            '''
            # set up a LocalCluster
            self.cluster = LocalCluster(n_workers=self.config['parallel']['n_workers'],
                                        threads_per_worker=self.config['parallel']['threads_per_worker'])
            # connect to local cluster
            self.c = Client(self.cluster)

            ''' set up processing queue '''
            self.q = OrderedSetQueue()

            # now we need to map our queue over task_runner and gather results in another queue.
            # user must setup specific tasks/jobs in task_runner, which (unfortunatelly)
            # cannot be defined inside a subclass -- only as a standalone function
            self.futures = self.c.map(task_runner, self.q, maxsize=self.config['parallel']['n_workers'])
            self.results = self.c.gather(self.futures)  # Gather results

            ''' DB connection is handled in subclass '''
            self.db = None

            ''' raw data are handled in subclass '''
            self.raw_data = None

        except Exception as e:
            print(e)
            traceback.print_exc()
            sys.exit()

    @staticmethod
    def get_config(_config_file):
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

    def set_up_logging(self, _name='archive', _mode='w'):
        """ Set up logging

            :param _name:
            :param _level: DEBUG, INFO, etc.
            :param _mode: overwrite log-file or append: w or a
            :return: logger instance
            """
        # 'debug', 'info', 'warning', 'error', or 'critical'
        if self.config['misc']['logging_level'] == 'debug':
            _level = logging.DEBUG
        elif self.config['misc']['logging_level'] == 'info':
            _level = logging.INFO
        elif self.config['misc']['logging_level'] == 'warning':
            _level = logging.WARNING
        elif self.config['misc']['logging_level'] == 'error':
            _level = logging.ERROR
        elif self.config['misc']['logging_level'] == 'critical':
            _level = logging.CRITICAL
        else:
            raise ValueError('Config file error: logging level must be ' +
                             '\'debug\', \'info\', \'warning\', \'error\', or \'critical\'')

        # get path to logs from config:
        _path = self.config['path']['path_logs']

        if not os.path.exists(_path):
            os.makedirs(_path)
        utc_now = datetime.datetime.utcnow()

        # http://www.blog.pythonlibrary.org/2012/08/02/python-101-an-intro-to-logging/
        _logger = logging.getLogger(_name)

        _logger.setLevel(_level)
        # create the logging file handler
        fh = logging.FileHandler(os.path.join(_path, '{:s}.{:s}.log'.format(_name, utc_now.strftime('%Y%m%d'))),
                                 mode=_mode)
        logging.Formatter.converter = time.gmtime

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # formatter = logging.Formatter('%(asctime)s %(message)s')
        fh.setFormatter(formatter)

        # add handler to logger object
        _logger.addHandler(fh)

        return _logger, utc_now.strftime('%Y%m%d')

    def shut_down_logger(self):
        """
            Prevent writing to multiple log-files after 'manual rollover'
        :return:
        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def check_logging(self):
        """
            Check if a new log file needs to be started and start it if necessary
        """
        if datetime.datetime.utcnow().strftime('%Y%m%d') != self.logger_utc_date:
            # reset
            self.shut_down_logger()
            self.logger, self.logger_utc_date = self.set_up_logging(_name='archive', _mode='a')

    def connect_to_db(self):
        """
            Connect to database. Specific details will differ for different users,
            so this should be implemented in subclass
        :return:
        """
        raise NotImplementedError

    def disconnect_from_db(self):
        """
            Disconnect from database. Specific details will differ for different users,
            so this should be implemented in subclass
        :return:
        """
        raise NotImplementedError

    def check_db_connection(self):
        """
            Check if DB connection is alive
        :return:
        """
        raise NotImplementedError

    def get_raw_data_descriptors(self):
        """
            Parse sources containing raw data and get high-level descriptors (like dates),
            by which raw data are sorted.
        :return:
        """
        raise NotImplementedError

    def cycle(self):
        """
            Main processing cycle to loop over dates and observations
        :return:
        """
        raise NotImplementedError

    def naptime(self):
        """
            Return time to sleep (in seconds) for archiving engine
            before waking up to rerun itself.
             During "working hours", it's set up in the config
             During nap time, it's nap_time_start_utc - utc_now()
        :return: time interval in seconds to sleep for
        """
        _config = self.config['misc']
        try:
            # local or UTC?
            tz = pytz.utc if _config['nap_time_frame'] == 'UTC' else None
            now = datetime.datetime.now(tz)

            if _config['nap_at_night']:

                last_midnight = datetime.datetime(now.year, now.month, now.day, tzinfo=tz)
                next_midnight = datetime.datetime(now.year, now.month, now.day, tzinfo=tz) \
                                + datetime.timedelta(days=1)

                hm_start = list(map(int, _config['nap_time_start'].split(':')))
                hm_stop = list(map(int, _config['nap_time_stop'].split(':')))

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
                        sleep_until = now + datetime.timedelta(minutes=_config['loop_interval'])
                        print('sleep until:', sleep_until)

                else:
                    h_after_midnight_start = hm_start[0] + hm_start[1] / 60.0
                    h_after_midnight_stop = hm_stop[0] + hm_stop[1] / 60.0

                    if (last_midnight + datetime.timedelta(hours=h_after_midnight_start) <
                            now < last_midnight + datetime.timedelta(hours=h_after_midnight_stop)):
                        sleep_until = last_midnight + datetime.timedelta(hours=h_after_midnight_stop)
                        print('sleep until:', sleep_until)
                    else:
                        sleep_until = now + datetime.timedelta(minutes=_config['loop_interval'])
                    print('sleep until:', sleep_until)

                return (sleep_until - now).total_seconds()

            else:
                # sleep for loop_interval minutes otherwise (return seconds)
                return _config['loop_interval'] * 60.0

        except Exception as _e:
            traceback.print_exc()
            self.logger.error('Failed to take a nap, taking a pill to fall asleep for an hour.')
            self.logger.error(_e)
            return 3600.0

    def sleep(self):
        """
            Take a nap in between cycles
        :return:
        """
        sleep_for = self.naptime()  # seconds
        # sleeping for longer than 10 minutes?
        if sleep_for > 10*60:
            self.logger.debug('Will disconnect from DB not to keep connection idle.')
            self.disconnect_from_db()
        self.logger.debug('Falling asleep for {:.1f} minutes.'.format(sleep_for / 60.0))
        time.sleep(sleep_for)


class RoboaoArchiver(Archiver):
    """
        A class representation of major data house-keeping/archiving tasks for
        Robo-AO, a robotic laser guide star adaptive optics system
    """
    def __init__(self, config_file=None):
        """
            Init
        :param config_file:
        """
        ''' initialize super class '''
        super(RoboaoArchiver, self).__init__(config_file=config_file)

        ''' connect to db: '''
        # will exit if this fails
        self.connect_to_db()

    def connect_to_db(self):
        """
            Connect to Robo-AO's MongoDB-powered database
        :return:
        """
        _config = self.config
        try:
            if self.logger is not None:
                self.logger.debug('Connecting to the Robo-AO database at {:s}:{:d}'.
                                  format(_config['database']['host'], _config['database']['port']))
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
            if self.logger is not None:
                self.logger.error(_e)
                self.logger.error('Failed to connect to the Robo-AO database at {:s}:{:d}'.
                                  format(_config['database']['host'], _config['database']['port']))
            # raise error
            raise ConnectionRefusedError
        try:
            # authenticate
            _db.authenticate(_config['database']['user'], _config['database']['pwd'])
            if self.logger is not None:
                self.logger.debug('Successfully authenticated with the Robo-AO database at {:s}:{:d}'.
                                  format(_config['database']['host'], _config['database']['port']))
        except Exception as _e:
            if self.logger is not None:
                self.logger.error(_e)
                self.logger.error('Authentication failed for the Robo-AO database at {:s}:{:d}'.
                                  format(_config['database']['host'], _config['database']['port']))
            raise ConnectionRefusedError
        try:
            # get collection with observations
            _coll_obs = _db[_config['database']['collection_obs']]
            if self.logger is not None:
                self.logger.debug('Using collection {:s} with obs data in the database'.
                                  format(_config['database']['collection_obs']))
        except Exception as _e:
            if self.logger is not None:
                self.logger.error(_e)
                self.logger.error('Failed to use a collection {:s} with obs data in the database'.
                                  format(_config['database']['collection_obs']))
            raise NameError
        try:
            # get collection with auxiliary stuff
            _coll_aux = _db[_config['database']['collection_aux']]
            if self.logger is not None:
                self.logger.debug('Using collection {:s} with aux data in the database'.
                                  format(_config['database']['collection_aux']))
        except Exception as _e:
            if self.logger is not None:
                self.logger.error(_e)
                self.logger.error('Failed to use a collection {:s} with aux data in the database'.
                                  format(_config['database']['collection_aux']))
            raise NameError
        try:
            # get collection with user access credentials
            _coll_usr = _db[_config['database']['collection_pwd']]
            if self.logger is not None:
                self.logger.debug('Using collection {:s} with user access credentials in the database'.
                                  format(_config['database']['collection_pwd']))
        except Exception as _e:
            if self.logger is not None:
                self.logger.error(_e)
                self.logger.error('Failed to use a collection {:s} with user access credentials in the database'.
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
            if self.logger is not None:
                self.logger.error(_e)

        if self.logger is not None:
            self.logger.debug('Successfully connected to Robo-AO database at {:s}:{:d}'.
                              format(_config['database']['host'], _config['database']['port']))

        # (re)define self.db
        self.db = dict()
        self.db['client'] = _client
        self.db['db'] = _db
        self.db['coll_obs'] = _coll_obs
        self.db['coll_aux'] = _coll_aux
        self.db['program_pi'] = _program_pi

    def disconnect_from_db(self):
        """
            Disconnect from Robo-AO's MongoDB database.
        :return:
        """
        self.logger.debug('Disconnecting from the database.')
        if self.db is not None:
            try:
                self.db['client'].close()
                self.logger.debug('Successfully disconnected from the database.')
            except Exception as e:
                self.logger.error('Failed to disconnect from the database.')
                self.logger.error(e)
            finally:
                # reset
                self.db = None
        else:
            self.logger.debug('No connection found.')

    def check_db_connection(self):
        """
            Check if DB connection is alive/established.
        :return: True if connection is OK
        """
        self.logger.debug('Checking database connection.')
        if self.db is None:
            try:
                self.connect_to_db()
            except Exception as e:
                print('Lost database connection.')
                self.logger.error('Lost database connection.')
                self.logger.error(e)
                return False
        else:
            try:
                # force connection on a request as the connect=True parameter of MongoClient seems
                # to be useless here
                self.db['client'].server_info()
            except pymongo.errors.ServerSelectionTimeoutError as e:
                print('Lost database connection.')
                self.logger.error('Lost database connection.')
                self.logger.error(e)
                return False

        return True

    def get_raw_data_descriptors(self):
        """
            Parse source(s) containing raw data and get dates with observational data.
        :return:
        """
        def is_date(_d, _fmt='%Y%m%d'):
            """
                Check if string (folder name) matches datetime format fmt
            :param _d:
            :param _fmt:
            :return:
            """
            try:
                datetime.datetime.strptime(_d, _fmt)
            except Exception as e:
                self.logger.error(e)
                return False
            return True

        # get all dates with some raw data from all input sources
        dates = dict()
        # Robo-AO's NAS archive contains folders named as YYYYMMDD.
        # Only consider data taken starting from archiving_start_date
        archiving_start_date = datetime.datetime.strptime(self.config['misc']['archiving_start_date'], '%Y/%m/%d')
        for _p in self.config['path']['path_raw']:
            dates[_p] = sorted([d for d in os.listdir(_p)
                                if os.path.isdir(os.path.join(_p, d))
                                and is_date(d, _fmt='%Y%m%d')
                                and datetime.datetime.strptime(d, '%Y%m%d') >= archiving_start_date
                                ])
        return dates

    def get_raw_data(self, _location, _date):
        """
            Get bzipped raw data file names at _location/_date
        :param _location:
        :param _date:
        :return:
        """
        return sorted([os.path.basename(_p) for _p in glob.glob(os.path.join(_location, _date, '*.fits.bz2'))])

    def cycle(self):
        """
            Main processing cycle
        :return:
        """
        try:
            while True:
                # check if a new log file needs to be started
                self.check_logging()

                # check if DB connection is alive/established
                connected = self.check_db_connection()

                if False:
                    # fake tasks:
                    tasks = [json.dumps({'task': 'test', 'a': aa}) for aa in range(20)]
                    tasks += [json.dumps({'task': 'bogus', 'a': aa}) for aa in range(20, 50)]

                    for task in tasks:
                        arch.q.put(task)

                if connected:
                    # get all dates with raw data for each raw data location
                    dates = self.get_raw_data_descriptors()
                    print(dates)

                    # iterate over data locations:
                    for location in dates:
                        for date in dates[location]:
                            self.logger.debug('Processing {:s} at {:s}'.format(date, location))
                            print(date)

                            # get raw data file names:
                            date_raw_data = self.get_raw_data(location, date)
                            if len(date_raw_data) == 0:
                                # no data? proceed to next date
                                self.logger.debug('No data found for {:s}'.format(date))
                                continue
                            else:
                                self.logger.debug('Found {:d} zipped fits-files for {:s}'.format(len(date_raw_data),
                                                                                                 date))
                            # print(date_raw_data)

                            # TODO: handle calibration data
                            try:
                                self.calibration(location, date, date_raw_data)
                            except Exception as _e:
                                print(_e)
                                traceback.print_exc()
                                self.logger.error(_e)
                                self.logger.error('Failed to process calibration data for {:s}.'.format(date))
                                continue

                            # TODO: handle auxiliary data

                            # TODO: get all observations

                            # TODO: iterate over individual observations


                            pass

                self.sleep()

        except KeyboardInterrupt:
            self.logger.error('User exited the archiver.')
            # try disconnecting from the database (if connected) and closing the cluster
            try:
                self.logger.info('Shutting down.')
                self.disconnect_from_db()
                self.cluster.close()
            finally:
                self.logger.info('Finished archiving cycle.')
                return False

    def calibration(self, _location, _date, _date_raw_data):
        """
            Handle calibration data

            Originally written by N. Law
        :param _location:
        :param _date:
        :param _date_raw_data:
        :return:
        """

        def sigma_clip_combine(img, out_fn, normalise=False, dark_bias_sub=False):
            """
                Combine all frames in FITS file img
            :param img:
            :param out_fn:
            :param normalise:
            :param dark_bias_sub:
            :return:
            """

            self.logger.debug('Making {:s}'.format(out_fn))
            # two passes:
            # 1/ generate average and RMS for each pixel
            # 2/ sigma-clipped average and RMS for each pixel
            # (i.e. only need to keep track of avg-sq and sq-avg arrays at any one time)
            sx = img[0].shape[1]
            sy = img[0].shape[0]

            avg = np.zeros((sy, sx), dtype=np.float32)
            avg_sq = np.zeros((sy, sx), dtype=np.float32)
            n_dps = np.zeros((sy, sx), dtype=np.float32)

            print("Image size:", sx, sy, len(img))
            self.logger.debug('Image size: {:d} {:d} {:d}'.format(sx, sy, len(img)))

            print("First pass")
            self.logger.debug("First pass")

            for i in img:
                avg += i.data
                avg_sq += i.data * i.data

            avg = avg / float(len(img))
            rms = np.sqrt((avg_sq / float(len(img))) - (avg * avg))

            # fits.PrimaryHDU(avg).writeto("avg.fits",clobber=True)
            # fits.PrimaryHDU(rms).writeto("rms.fits",clobber=True)

            sigma_clip_avg = np.zeros((sy, sx), dtype=np.float32)
            sigma_clip_n = np.zeros((sy, sx), dtype=np.float32)

            print("Second pass")
            for i in img:
                sigma_mask = np.fabs((np.array(i.data, dtype=np.float32) - avg) / rms)

                sigma_mask[sigma_mask > 3.0] = 100
                sigma_mask[sigma_mask <= 1.0] = 1
                sigma_mask[sigma_mask == 100] = 0

                sigma_clip_avg += i.data * sigma_mask
                sigma_clip_n += sigma_mask

            sigma_clip_avg /= sigma_clip_n

            # set the average flat level to 1.0
            if normalise:
                sigma_clip_avg /= np.average(sigma_clip_avg[np.isfinite(sigma_clip_avg)])

            if dark_bias_sub:
                sigma_clip_avg -= np.average(sigma_clip_avg[sy - 50:sy, sx - 50:sx])

            fits.PrimaryHDU(sigma_clip_avg).writeto(out_fn, clobber=True)

        # path to zipped raw files
        path_date = os.path.join(_location, _date)

        # get calibration file names:
        pattern_fits = r'.fits.bz2\Z'
        bias = [re.split(pattern_fits, s)[0] for s in _date_raw_data if re.match('bias_', s) is not None]
        flat = [re.split(pattern_fits, s)[0] for s in _date_raw_data if re.match('flat_', s) is not None]
        dark = [re.split(pattern_fits, s)[0] for s in _date_raw_data if re.match('dark_', s) is not None]

        # TODO: check in database if needs to be (re)done


        n_darks = len(dark)
        n_flats = len(flat)

        if n_darks > 9 and n_flats > 4:

            # output dir
            _path_out = os.path.join(self.config['path']['path_archive'], _date, 'calib')
            if not os.path.exists(os.path.join(_path_out)):
                os.makedirs(os.path.join(_path_out))

            # find the combine the darks:
            for d in dark:
                # file name with extension
                fz = '{:s}.fits.bz2'.format(d)
                f = '{:s}.fits'.format(d)
                camera_mode = f.split('_')[1]
                if camera_mode != '0':
                    # mode 0 is handled below

                    # unzip:
                    lbunzip2(_path_in=path_date, _files=fz, _path_out=self.config['path']['path_tmp'], _keep=True)

                    unzipped = os.path.join(self.config['path']['path_tmp'], f)

                    out_fn = os.path.join(_path_out, 'dark_{:s}.fits'.format(camera_mode))
                    with fits.open(unzipped) as img:
                        sigma_clip_combine(img, out_fn, dark_bias_sub=True)

                    # clean up after yoself!
                    try:
                        os.remove(unzipped)
                    except Exception as _e:
                        self.logger.error(_e)

            # generate the flats' dark
            # those files with mode '0' are the relevant ones:
            fz = ['{:s}.fits.bz2'.format(d) for d in dark if d.split('_')[1] == '0']
            f = ['{:s}.fits'.format(d) for d in dark if d.split('_')[1] == '0']
            unzipped = [os.path.join(self.config['path']['path_tmp'], _f) for _f in f]

            # unzip:
            lbunzip2(_path_in=path_date, _files=fz, _path_out=self.config['path']['path_tmp'], _keep=True)

            img = []
            for uz in unzipped:
                img.append(fits.open(uz)[0])
                # clean up after yoself!
                try:
                    os.remove(uz)
                except Exception as _e:
                    self.logger.error(_e)

            flat_dark_fn = os.path.join(_path_out, 'dark_0.fits')
            sigma_clip_combine(img, flat_dark_fn, dark_bias_sub=False)

            with fits.open(flat_dark_fn) as fdn:
                flat_dark = np.array(fdn[0].data, dtype=np.float32)

            # make the flats:
            for filt in ["Sg", "Si", "Sr", "Sz", "lp600", "c"]:
                print("Making {:s} flat".format(filt))
                flats = []

                fz = ['{:s}.fits.bz2'.format(_f) for _f in flat if _f.split('_')[2] == filt]
                f = ['{:s}.fits'.format(_f) for _f in flat if _f.split('_')[2] == filt]
                unzipped = [os.path.join(self.config['path']['path_tmp'], _f) for _f in f]

                # unzip:
                lbunzip2(_path_in=path_date, _files=fz, _path_out=self.config['path']['path_tmp'], _keep=True)

                for uz in unzipped:

                    flt = fits.open(uz)[0]
                    flt.data = np.array(flt.data, dtype=np.float32)
                    flt.data -= flat_dark
                    # clean up after yoself!
                    try:
                        os.remove(uz)
                    except Exception as _e:
                        self.logger.error(_e)

                    flats.append(flt)

                out_fn = os.path.join(_path_out, 'flat_{:s}.fits'.format(filt))
                sigma_clip_combine(flats, out_fn, normalise=True)

        else:
            raise RuntimeError('No enough calibration files for {:s}'.format(_date))


def lbunzip2(_path_in, _files, _path_out, _keep=True, _v=False):

    """
        A wrapper around lbunzip2 - a parallel version of bunzip2
    :param _path_in: folder with the files to be unzipped
    :param _files: string or list of strings with file names to be uncompressed
    :param _path_out: folder to place the output
    :param _keep: keep the original?
    :return:
    """

    try:
        p0 = subprocess.Popen(['lbunzip2'])
        p0.wait()
    except Exception as _e:
        print(_e)
        print('lbzip2 not installed in the system. go ahead and install it!')
        return False

    if isinstance(_files, str):
        _files_list = [_files]
    else:
        _files_list = _files

    files_size = sum([os.stat(os.path.join(_path_in, fs)).st_size for fs in _files_list])
    # print(files_size)

    if _v:
        bar = pyprind.ProgBar(files_size, stream=1, title='Unzipping files', monitor=True)
    for _file in _files_list:
        file_in = os.path.join(_path_in, _file)
        file_out = os.path.join(_path_out, os.path.splitext(_file)[0])
        if os.path.exists(file_out) and os.stat(file_out).st_size != 0:
            # print('uncompressed file {:s} already exists, skipping'.format(file_in))
            if _v:
                bar.update(iterations=os.stat(file_in).st_size)
            continue
        # else go ahead
        # print('lbunzip2 <{:s} >{:s}'.format(file_in, file_out))
        with open(file_in, 'rb') as _f_in, open(file_out, 'wb') as _f_out:
            _p = subprocess.Popen('lbunzip2'.split(), stdin=subprocess.PIPE, stdout=_f_out)
            _p.communicate(input=_f_in.read())
            # wait for it to finish
            _p.wait()
        # remove the original if requested:
        if not _keep:
            _p = subprocess.Popen(['rm', '-f', '{:s}'.format(os.path.join(_path_in, _file))])
            # wait for it to finish
            _p.wait()
            # pass
        if _v:
            bar.update(iterations=os.stat(file_in).st_size)

    return True


def task_runner(argdict):
    """
        Helper function that maps over 'data'

    :param argdict: json-dumped dictionary with (named) parameters for the task.
                    must contain 'task' key with the task name known to this helper function
            json.dumps is used to convert the dict to a hashable type - string - so that
            it can be used with SetQueue or OrderedSetQueue. the latter two are in turn
            used instead of regular queues to be able to check if a task has been enqueued already
    :return:
    """
    try:
        # unpack jsonified dict:
        argdict = json.loads(argdict)

        # assert 'task' in argdict, 'specify which task to run'

        print('running task {:s}'.format(argdict['task']))

        if argdict['task'] == 'test':
            out = job_test(argdict['a'])
            print(out)

        elif argdict['task'] == 'bogus':
            out = job_bogus(argdict['a'])
            print(out)

        return True

    except Exception as e:
        print('task failed saying \"{:s}\"'.format(str(e)))
        # traceback.print_exc()
        return True


def job_test(a):
    for _i in range(200):
        for _j in range(100):
            for k in range(500):
                a += 3 ** 2
    return a


def job_bogus(a):
    for _i in range(200):
        for _j in range(100):
            for k in range(500):
                a += 4 ** 2
    return a


# TODO:
class Observation(object):
    pass


# TODO:
class Pipeline(object):
    pass


if __name__ == '__main__':

    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Manage data archive for Robo-AO')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)

    args = parser.parse_args()

    # init archiver:
    arch = RoboaoArchiver(args.config_file)

    # start the archiver main house-keeping cycle:
    arch.cycle()
