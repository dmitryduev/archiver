import argparse
import signal
from collections import OrderedDict
import pytz
from distributed import Client, LocalCluster
import threading
from queue import Queue
import time
import datetime
import inspect
import os
import shutil
import subprocess
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import glob
import traceback
import sys
import json
import logging
import collections
import pymongo
from bson import json_util
import re
from astropy.io import fits
import pyprind
import functools
import hashlib


def export_fits(path, _data, _header=None):
    """
        Save fits file overwriting if exists
    :param path:
    :param _data:
    :param _header:
    :return:
    """
    if _header is not None:
        hdu = fits.PrimaryHDU(_data, header=_header)
    else:
        hdu = fits.PrimaryHDU(_data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(path, overwrite=True)


def get_fits_header(fits_file):
    """
        Get fits-file header
    :param fits_file:
    :return:
    """
    # read fits:
    with fits.open(os.path.join(fits_file)) as hdulist:
        # header:
        header = OrderedDict()
        for _entry in hdulist[0].header.cards:
            header[_entry[0]] = _entry[1:]

    return header


def sigma_clip(x, sigma, niter):
    x = np.array(x)
    if len(x) > 3:
        for i in range(niter):
            xt = x - np.mean(x)
            x = x[np.where(abs(xt) < sigma*np.std(xt))]
    return list(x)


def lbunzip2(_path_in, _files, _path_out, _keep=True, _rewrite=True, _v=False):

    """
        A wrapper around lbunzip2 - a parallel version of bunzip2
    :param _path_in: folder with the files to be unzipped
    :param _files: string or list of strings with file names to be uncompressed
    :param _path_out: folder to place the output
    :param _rewrite: rewrite if output exists?
    :param _keep: keep the original?
    :param _v: verbose?
    :return:
    """

    try:
        subprocess.run(['which', 'lbunzip2'], check=True)
        print('found lbzip2 in the system')
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

        if not _rewrite:
            if os.path.exists(file_out) and os.stat(file_out).st_size != 0:
                # print('uncompressed file {:s} already exists, skipping'.format(file_in))
                if _v:
                    bar.update(iterations=os.stat(file_in).st_size)
                continue
        # else go ahead
        # print('lbunzip2 <{:s} >{:s}'.format(file_in, file_out))
        with open(file_in, 'rb') as _f_in, open(file_out, 'wb') as _f_out:
            subprocess.run(['lbunzip2'], input=_f_in.read(), stdout=_f_out)
        # remove the original if requested:
        if not _keep:
            subprocess.run(['rm', '-f', '{:s}'.format(os.path.join(_path_in, _file))], check=True)
        if _v:
            bar.update(iterations=os.stat(file_in).st_size)

    return True


def memoize(f):
    """ Minimalistic memoization decorator.
    http://code.activestate.com/recipes/577219-minimalistic-memoization/ """

    cache = {}

    @functools.wraps(f)
    def memf(*x):
        if x not in cache:
            cache[x] = f(*x)
        return cache[x]
    return memf


def mdate_walk(_path):
    """
        Inspect directory tree contents to get max mdate of all files within it
    :param _path:
    :return:
    """
    if not os.path.exists(_path):
        return utc_now()
    # modified time for the parent folder:
    mtime = datetime.datetime.utcfromtimestamp(os.stat(_path).st_mtime)
    # print(mtime)
    for root, _, files in os.walk(_path, topdown=False):
        # only check the files:
        for _f in files:
            path_f = os.path.join(root, _f)
            mtime_f = datetime.datetime.utcfromtimestamp(os.stat(path_f).st_mtime)
            if mtime_f > mtime:
                mtime = mtime_f
            # print(path_f, mtime_f)
        # don't actually need to check dirs
        # for _d in dirs:
        #     print(os.path.join(root, _d))
    return mtime


def utc_now():
    return datetime.datetime.now(pytz.utc)


class TimeoutError(Exception):
    def __init__(self, value='Operation timed out'):
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
        new_f.__name__ = f.__name__
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

        # print('_______________________')
        # print(item in self.queue)
        # # print(item)
        # for ii, i in enumerate(self.queue):
        #     print(ii, i['id'], i['task'])
        # print('_______________________')

        if item in self.queue:
            # do not count it since it's not going to be added to the set!
            # see line 144 in queue.py (python 3.6)
            self.unfinished_tasks -= 1
        self.queue.add(item)

    def _get(self):
        self.unfinished_tasks = max(self.unfinished_tasks - 1, 0)
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

            # make dirs if necessary:
            for _pp in ('archive', 'tmp'):
                _path = self.config['path']['path_{:s}'.format(_pp)]
                if not os.path.exists(_path):
                    os.makedirs(_path)
                    self.logger.debug('Created {:s}'.format(_path))

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
            # we will be submitting processing tasks to it
            # self.q = OrderedSetQueue()
            # self.q = SetQueue()
            self.q = Queue()

            # keep hash values of enqueued tasks to prevent submitting particular task multiple times
            self.task_hashes = set()

            # now we need to map our queue over task_runner and gather results in another queue.
            # user must setup specific tasks/jobs in task_runner, which (unfortunately)
            # cannot be defined inside a subclass -- only as a standalone function
            self.futures = self.c.map(self.task_runner, self.q, maxsize=self.config['parallel']['n_workers'])
            self.results = self.c.gather(self.futures)  # Gather results

            self.logger.debug('Successfully set up dask.distributed cluster')

            # Pipelining tasks (dict of form {'task': 'task_name', 'param_a': param_a_value}, jsonified)
            # to distributed queue for execution as self.q.put(task)

            # note: result harvester is defined and started in subclass!

            ''' DB connection is handled in subclass '''
            self.db = None

            ''' raw data are handled in subclass '''
            self.raw_data = None

        except Exception as e:
            print(e)
            traceback.print_exc()
            sys.exit()

    def hash_task(self, _task):
        """
            Compute hash for a hashable task
        :return:
        """
        ht = hashlib.blake2b(digest_size=12)
        ht.update(_task.encode('utf-8'))
        hsh = ht.hexdigest()
        # # it's a set, so don't worry about adding a hash multiple times
        # self.task_hashes.add(hsh)

        return hsh

    # def unhash_task(self, _hsh):
    #     """
    #         Remove hexdigest-ed hash from self.task_hashes
    #     :return:
    #     """
    #     self.task_hashes.remove(_hsh)

    def harvester(self):
        """
            Harvest processing results from dask.distributed results queue, update DB entries if necessary.
            Specific implementation details are defined in subclass
        :return:
        """
        raise NotImplementedError

    @staticmethod
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
        raise NotImplementedError

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

        ''' start results harvester in separate thread '''
        self.running = True
        self.h = threading.Thread(target=self.harvester)
        self.h.start()

    def harvester(self):
        """
            Harvest processing results from dask.distributed results queue, update DB entries if necessary
        :return:
        """
        # make sure the archiver is running. this is to protect from frozen thread on (user) exit
        while self.running:
            # get new results from queue one by one
            while not self.results.empty():
                try:
                    result = self.results.get()
                    # self.logger.debug('Task finished saying: {:s}'.format(str(result)))
                    print('Task finished saying:\n', str(result))
                    self.logger.info('Task {:s} for {:s} finished with status {:s}'.format(result['job'],
                                                                                           result['_id'],
                                                                                           result['status']))
                    # TODO: update DB entry
                    if 'db_record_update' in result:
                        self.update_db_entry(_collection='coll_obs', upd=result['db_record_update'])
                    # TODO: remove from self.task_hashes
                    if result['status'] == 'ok' and 'hash' in result:
                        self.task_hashes.remove(result['hash'])
                except Exception as _e:
                    print(_e)
                    traceback.print_exc()
                    self.logger.error(_e)
            # don't need to check that too often
            time.sleep(5)

    @staticmethod
    def task_runner(argdict_and_hash):
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
            # unpack jsonified dict representing task:
            argdict = json.loads(argdict_and_hash[0], object_hook=json_util.object_hook)
            # get task hash:
            _task_hash = argdict_and_hash[1]

            assert 'task' in argdict, 'specify which task to run'
            print('running task {:s}'.format(argdict['task']))

            if argdict['task'] == 'bright_star_pipeline':
                result = job_bright_star_pipeline(_id=argdict['id'], _config=argdict['config'],
                                                  _db_entry=argdict['db_entry'], _task_hash=_task_hash)

            elif argdict['task'] == 'bright_star_pipeline:strehl':
                result = {'status': 'error', 'message': 'not implemented yet'}

            elif argdict['task'] == 'bright_star_pipeline:pca':
                result = {'status': 'error', 'message': 'not implemented yet'}

            elif argdict['task'] == 'faint_star_pipeline':
                result = {'status': 'error', 'message': 'not implemented yet'}

            elif argdict['task'] == 'faint_star_pipeline:strehl':
                result = {'status': 'error', 'message': 'not implemented yet'}

            elif argdict['task'] == 'faint_star_pipeline:pca':
                result = {'status': 'error', 'message': 'not implemented yet'}

            elif argdict['task'] == 'extended_object_pipeline':
                result = {'status': 'error', 'message': 'not implemented yet'}

            else:
                result = {'status': 'error', 'message': 'unknown task'}

        except Exception as _e:
            # exception here means bad argdict.
            print(_e)
            traceback.print_exc()
            result = {'status': 'error', 'message': str(_e)}

        return result

    @timeout(seconds_before_timeout=60)
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

    @timeout(seconds_before_timeout=60)
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

    @timeout(seconds_before_timeout=60)
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

    @timeout(seconds_before_timeout=60)
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

    @staticmethod
    def get_raw_data(_location, _date):
        """
            Get bzipped raw data file names at _location/_date
        :param _location:
        :param _date:
        :return:
        """
        return sorted([os.path.basename(_p) for _p in glob.glob(os.path.join(_location, _date, '*.fits.bz2'))])

    @timeout(seconds_before_timeout=30)
    def insert_db_entry(self, _collection=None, _db_entry=None):
        """
            Insert a document _doc to collection _collection in DB.
            It is monitored for timeout in case DB connection hangs for some reason
        :param _collection:
        :param _db_entry:
        :return:
        """
        assert _collection is not None, 'Must specify collection'
        assert _db_entry is not None, 'Must specify document'
        try:
            self.db[_collection].insert_one(_db_entry)
        except Exception as e:
            self.logger.error('Error inserting {:s} into {:s}'.format(_db_entry, _collection))
            self.logger.error(e)

    # @timeout(seconds_before_timeout=30)
    def update_db_entry(self, _collection=None, upd=None):
        """
            Update DB entry
            Note: it's mainly used by archiver's harvester, which is run in separate thread,
                  therefore signals don't work :(
        :return:
        """
        assert _collection is not None, 'Must specify collection'
        assert upd is not None, 'Must specify update'
        try:
            self.db[_collection].update_one(upd[0], upd[1])
            self.logger.info('Updated DB entry for {:s}'.format(upd[0]['_id']))
        except Exception as e:
            self.logger.error('Error executing {:s} for {:s}'.format(upd, _collection))
            self.logger.error(e)

    @staticmethod
    def empty_db_aux_entry(_date):
        """
                A dummy database record for a science observation
            :param: _date YYYYMMDD which serves as _id
            :return:
            """
        time_now_utc = utc_now()
        return {
            '_id': _date,
            'calib': {'done': False,
                      'raw': {'bias': [],
                              'dark': [],
                              'flat': []},
                      'retries': 0,
                      'last_modified': time_now_utc},
            'seeing': {'done': False,
                       'frames': [],
                       'retries': 0,
                       'last_modified': time_now_utc},
            'contrast_curve': {'done': False,
                               'retries': 0,
                               'last_modified': time_now_utc},
            'strehl': {'done': False,
                       'retries': 0,
                       'last_modified': time_now_utc}
               }

    def cycle(self):
        """
            Main processing cycle
        :return:
        """
        try:
            # set up patterns, as these are not going to change
            # check the endings (\Z) and skip _N.fits.bz2:
            # science obs must start with program number (e.g. 24_ or 24.1_)
            pattern_start = r'\d+.?\d??_'
            # must be a bzipped fits file
            pattern_end = r'.[0-9]{6}.fits.bz2\Z'
            pattern_fits = r'.fits.bz2\Z'

            while True:

                # check if a new log file needs to be started
                self.check_logging()

                # check if DB connection is alive/established
                connected = self.check_db_connection()

                if connected:

                    # get all dates with raw data for each raw data location
                    dates = self.get_raw_data_descriptors()
                    print(dates)

                    # iterate over data locations:
                    for location in dates:
                        for date in dates[location]:
                            # Each individual step where I know something could go wrong is placed inside a try-except
                            # clause. Everything else is captured outside causing the main while loop to terminate.
                            self.logger.debug('Processing {:s} at {:s}'.format(date, location))
                            print(date)

                            # get all raw data file names for the date, including calibration, seeing, and pointing:
                            try:
                                date_raw_data = self.get_raw_data(location, date)
                                if len(date_raw_data) == 0:
                                    # no data? proceed to next date
                                    self.logger.debug('No data found for {:s}'.format(date))
                                    continue
                                else:
                                    self.logger.debug('Found {:d} zipped fits-files for {:s}'.format(len(date_raw_data),
                                                                                                     date))
                            except Exception as _e:
                                print(_e)
                                traceback.print_exc()
                                self.logger.error(_e)
                                self.logger.error('Failed to get all raw data file names for {:s}.'.format(date))
                                continue

                            ''' auxiliary data '''
                            # look up aux entry for date in the database:
                            try:
                                select = self.db['coll_aux'].find_one({'_id': date}, max_time_ms=5000)
                                # if entry not in database, create empty one and populate it
                                if select is None:
                                    # insert empty entry for date into aux database:
                                    self.insert_db_entry(_collection='coll_aux',
                                                         _db_entry=self.empty_db_aux_entry(date))
                            except Exception as _e:
                                print(_e)
                                traceback.print_exc()
                                self.logger.error(_e)
                                self.logger.error('Error in handling aux database entry for {:s}.'.format(date))
                                continue

                            # handle calibration data
                            try:
                                # we do this in a serial way, i.e. before proceeding with everything else
                                # because calibration data are needed by everything else
                                self.calibration(location, date, date_raw_data)
                            except Exception as _e:
                                print(_e)
                                # traceback.print_exc()
                                self.logger.error(_e)
                                self.logger.error('Failed to process calibration data for {:s}.'.format(date))
                                continue

                            # TODO: handle auxiliary data (seeing, summary contrast curves and Strehl ratios)

                            # once done with aux data processing, get entry from aux collection in DB:
                            # look up aux entry for date in the database:
                            try:
                                aux_date = self.db['coll_aux'].find_one({'_id': date}, max_time_ms=5000)
                            except Exception as _e:
                                print(_e)
                                traceback.print_exc()
                                self.logger.error(_e)
                                self.logger.error('Error in handling aux database entry for {:s}.'.format(date))
                                continue

                            ''' science data '''
                            # get all science observations
                            try:
                                # skip calibration files and pointings
                                date_obs = [re.split(pattern_fits, s)[0] for s in date_raw_data
                                            if re.search(pattern_end, s) is not None and
                                            re.match(pattern_start, s) is not None and
                                            re.match('bias_', s) is None and
                                            re.match('dark_', s) is None and
                                            re.match('flat_', s) is None and
                                            re.match('pointing_', s) is None and
                                            re.match('seeing_', s) is None]
                                print(date_obs)
                                if len(date_obs) == 0:
                                    # no data? proceed to next date
                                    self.logger.info('No science data found for {:s}'.format(date))
                                    continue
                                else:
                                    self.logger.debug(
                                        'Found {:d} zipped science fits-files for {:s}'.format(len(date_raw_data),
                                                                                               date))
                            except Exception as _e:
                                print(_e)
                                traceback.print_exc()
                                self.logger.error(_e)
                                self.logger.error('Failed to get all raw science data file names for {:s}.'
                                                  .format(date))
                                continue

                            # TODO: iterate over individual observations
                            for obs in date_obs:
                                try:
                                    # look up entry for obs in DB:
                                    select = self.db['coll_obs'].find_one({'_id': obs}, max_time_ms=10000)

                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error('Failed to look up entry for {:s} in DB.'.format(obs))
                                    self.logger.error(_e)
                                    continue

                                try:
                                    # init RoboaoObservation object
                                    roboao_obs = RoboaoObservation(_id=obs, _aux=aux_date,
                                                                   _program_pi=self.db['program_pi'],
                                                                   _db_entry=select,
                                                                   _config=self.config)
                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error('Failed to set up obs object for {:s}.'.format(obs))
                                    self.logger.error(_e)
                                    continue

                                try:
                                    # init DB entry if not in DB
                                    if select is None:
                                        self.insert_db_entry(_collection='coll_obs', _db_entry=roboao_obs.db_entry)
                                        self.logger.info('Inserted {:s} into DB'.format(obs))
                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error(_e)
                                    self.logger.error('Initial DB insertion failed for {:s}.'.format(obs))
                                    continue

                                try:
                                    # check raws
                                    s = roboao_obs.check_raws(_location=location, _date=date,
                                                              _date_raw_data=date_raw_data)
                                    # changes detected?
                                    if s['status'] == 'ok' and s['message'] is not None:
                                        print(s['db_record_update'])
                                        self.update_db_entry(_collection='coll_obs', upd=s['db_record_update'])
                                        self.logger.info('Corrected raw_data entry for {:s}'.format(obs))
                                        self.logger.debug(json.dumps(s['db_record_update'], default=json_util.default))
                                    # something failed?
                                    elif s['status'] == 'error':
                                        self.logger.error('{:s}, checking raw files failed: {:s}'.format(obs,
                                                                                                         s['message']))
                                        # proceed to next obs:
                                        continue
                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error(_e)
                                    self.logger.error('Raw files check failed for {:s}.'.format(obs))
                                    continue

                                try:
                                    # check that DB entry reflects reality
                                    s = roboao_obs.check_db_entry()
                                    # discrepancy detected?
                                    if s['status'] == 'ok' and s['message'] is not None:
                                        self.update_db_entry(_collection='coll_obs', upd=s['db_record_update'])
                                        self.logger.info('Corrected DB entry for {:s}'.format(obs))
                                        self.logger.debug(json.dumps(s['db_record_update'], default=json_util.default))
                                    # something failed?
                                    elif s['status'] == 'error':
                                        self.logger.error('{:s}, checking DB entry failed: {:s}'.format(obs,
                                                                                                         s['message']))
                                        # proceed to next obs:
                                        continue
                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error(_e)
                                    self.logger.error('Raw files check failed for {:s}.'.format(obs))
                                    continue

                                try:
                                    # we'll be adding one task per observation at a time to avoid
                                    # complicating things.
                                    # self.task_runner will take care of executing the task
                                    pipe_task = roboao_obs.get_task()

                                    if pipe_task is not None:
                                        # try enqueueing. self.task_hashes takes care of possible duplicates
                                        # use json dumps to serialize input dictionary _task. this way,
                                        # it may be pickled and enqueued (and also hashed:)
                                        pipe_task_hashable = json.dumps(pipe_task, default=json_util.default)

                                        # compute hash for task:
                                        pipe_task_hash = self.hash_task(pipe_task_hashable)
                                        # not enqueued?
                                        if pipe_task_hash not in self.task_hashes:
                                            print({'id': pipe_task['id'], 'task': pipe_task['task']})
                                            # mark as enqueued in DB:
                                            # self.update_db_entry(_collection='coll_obs',
                                            #                      upd=pipe_task['db_record_update'])
                                            # enqueue the task together with its hash:
                                            self.q.put((pipe_task_hashable, pipe_task_hash))
                                            # bookkeeping:
                                            self.task_hashes.add(pipe_task_hash)
                                            self.logger.info('Enqueueing task {:s} for {:s}'.format(pipe_task['task'],
                                                                                                    pipe_task['id']))
                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error(_e)
                                    self.logger.error('Failed to get/enqueue a task for {:s}.'.format(obs))
                                    continue

                                try:
                                    # check distribution status
                                    s = roboao_obs.check_distributed()
                                    if s['status'] == 'ok' and s['message'] is not None:
                                        self.update_db_entry(_collection='coll_obs', upd=s['db_record_update'])
                                        self.logger.info('updated distribution status for {:s}'.format(obs))
                                        self.logger.debug(json.dumps(s['db_record_update'], default=json_util.default))
                                    # something failed?
                                    elif s['status'] == 'error':
                                        self.logger.error('{:s}, checking distribution status failed: {:s}'.format(obs,
                                                                                                         s['message']))
                                        # proceed to next obs:
                                        continue
                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error(_e)
                                    self.logger.error('Raw files check failed for {:s}.'.format(obs))
                                    continue

                # unfinished tasks live here:
                # print(self.q.unfinished_tasks)
                # released tasks live here:
                # print(self.c.scheduler.released())
                self.sleep()

        except KeyboardInterrupt:
            # user ctrl-c'ed
            self.running = False
            self.logger.error('User exited the archiver.')
            # try disconnecting from the database (if connected) and closing the cluster
            try:
                self.logger.info('Shutting down.')
                self.logger.debug('Cleaning tmp directory.')
                shutil.rmtree(self.config['path']['path_tmp'])
                os.makedirs(self.config['path']['path_tmp'])
                self.logger.debug('Disconnecting from DB.')
                self.disconnect_from_db()
                self.logger.debug('Shutting down dask.distributed cluster.')
                self.cluster.close()
            finally:
                self.logger.info('Finished archiving cycle.')
                return False

        except RuntimeError as e:
            # any other error not captured otherwise
            print(e)
            traceback.print_exc()
            self.logger.error(e)
            self.logger.error('Unknown error, exiting. Please check the logs.')
            self.running = False
            try:
                self.logger.info('Shutting down.')
                self.logger.debug('Cleaning tmp directory.')
                shutil.rmtree(self.config['path']['path_tmp'])
                os.makedirs(self.config['path']['path_tmp'])
                self.logger.debug('Disconnecting from DB.')
                self.disconnect_from_db()
                self.logger.debug('Shutting down dask.distributed cluster.')
                self.cluster.close()
            finally:
                self.logger.info('Finished archiving cycle.')
                return False

    @timeout(seconds_before_timeout=60)
    def load_darks_and_flats(self, _date, _mode, _filt, image_size_x=1024):
        """
            Load darks and flats
        :param _date:
        :param _mode:
        :param _filt:
        :param image_size_x:
        :return:
        """
        try:
            _path_calib = os.path.join(self.config['path_archive', _date, 'calib'])
            if image_size_x == 256:
                dark_image = os.path.join(_path_calib, 'dark_{:s}4.fits'.format(str(_mode)))
            else:
                dark_image = os.path.join(_path_calib, 'dark_{:s}.fits'.format(str(_mode)))
            flat_image = os.path.join(_path_calib, 'flat_{:s}.fits'.format(_filt))

            if not os.path.exists(dark_image) or not os.path.exists(flat_image):
                return None, None
            else:
                with fits.open(dark_image) as dark, fits.open(flat_image) as flat:
                    # replace NaNs if necessary
                    if image_size_x == 256:
                        return np.nan_to_num(dark[0].data), np.nan_to_num(flat[0].data[384:640, 384:640])
                    else:
                        return np.nan_to_num(dark[0].data), np.nan_to_num(flat[0].data)
        except RuntimeError:
            # Failed? Make sure to mark calib.done in DB as False!
            self.db['coll_aux'].update_one(
                {'_id': _date},
                {
                    '$set': {
                        'calib.done': False,
                        'calib.raw.flat': [],
                        'calib.raw.dark': [],
                        'calib.last_modified': utc_now()
                    }
                }
            )

    @timeout(seconds_before_timeout=600)
    def calibration(self, _location, _date, _date_raw_data):
        """
            Handle calibration data

            It is monitored for timout

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

            # fits.PrimaryHDU(avg).writeto("avg.fits", overwrite=True)
            # fits.PrimaryHDU(rms).writeto("rms.fits", overwrite=True)

            sigma_clip_avg = np.zeros((sy, sx), dtype=np.float32)
            sigma_clip_n = np.zeros((sy, sx), dtype=np.float32)

            print("Second pass")
            self.logger.debug("Second pass")
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

            fits.PrimaryHDU(sigma_clip_avg).writeto(out_fn, overwrite=True)
            self.logger.debug("Successfully made {:s}".format(out_fn))

        # path to zipped raw files
        path_date = os.path.join(_location, _date)

        # output dir
        _path_out = os.path.join(self.config['path']['path_archive'], _date, 'calib')

        # get calibration file names:
        pattern_fits = r'.fits.bz2\Z'
        bias = [re.split(pattern_fits, s)[0] for s in _date_raw_data if re.match('bias_', s) is not None]
        flat = [re.split(pattern_fits, s)[0] for s in _date_raw_data if re.match('flat_', s) is not None]
        dark = [re.split(pattern_fits, s)[0] for s in _date_raw_data if re.match('dark_', s) is not None]

        # check in database if needs to be (re)done
        _select = self.db['coll_aux'].find_one({'_id': _date})

        # FIXME: this is not to break older entries. remove once production ready
        if 'calib' not in _select:
            _select['calib'] = self.empty_db_aux_entry(None)['calib']

        # check folder modified date:
        time_tag = mdate_walk(_path_out)

        # not done or files changed?
        if (not _select['calib']['done']) or \
                (set(_select['calib']['raw']['flat']) != set(flat)) or \
                (set(_select['calib']['raw']['dark']) != set(dark)) or \
                (time_tag - _select['calib']['last_modified']).total_seconds() > 1.0:
            make_calib = True
        else:
            make_calib = False

        if make_calib:
            n_darks = len(dark)
            n_flats = len(flat)

            # enough data to make master calibration files?
            if n_darks > 9 and n_flats > 4:
                # update DB entry:
                self.db['coll_aux'].update_one(
                    {'_id': _date},
                    {
                        '$set': {
                            'calib.done': False,
                            'calib.raw.flat': flat,
                            'calib.raw.dark': dark,
                            'calib.last_modified': time_tag
                        }
                    }
                )

                # output dir exists?
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

                # success!
                # check new folder modified date:
                time_tag = mdate_walk(_path_out)
                # update DB entry:
                self.db['coll_aux'].update_one(
                    {'_id': _date},
                    {
                        '$set': {
                            'calib.done': True,
                            'calib.last_modified': time_tag
                        }
                    }
                )

            else:
                raise RuntimeError('No enough calibration files for {:s}'.format(_date))


class Observation(object):
    """
        This is mainly to show the tentative structure for future use cases
    """
    def __init__(self, _id=None, _aux=None):
        """
            Initialize Observation object
        :param _id:
        :param _aux:
        :return:
        """
        assert _id is not None, 'Must specify unique obs id'
        # obs unique id:
        self.id = _id

        self.aux = _aux

    def parse(self, **kwargs):
        """
            Parse obs info (e.g. contained in id, or fits header) to be injected into DB
            Define decision chain for observation pipelining
        :return:
        """
        raise NotImplementedError

    def init_db_entry(self):
        """
            Initialize DB entry
        :return:
        """
        raise NotImplementedError

    def check_db_entry(self):
        """
            Check if DB entry reflects reality
        :return:
        """
        raise NotImplementedError

    def get_task(self, **kwargs):
        """
            Construct decision chain
        :return:
        """
        raise NotImplementedError


class RoboaoObservation(Observation):
    def __init__(self, _id=None, _aux=None, _program_pi=None, _db_entry=None, _config=None):
        """
            Initialize Observation object
        :param _id:
        :param _aux:
        :return:
        """
        ''' initialize super class '''
        super(RoboaoObservation, self).__init__(_id=_id, _aux=_aux)

        # current DB entry
        if _db_entry is None:
            # db entry does not exist?
            # parse obs name
            _obs_info = self.parse(_program_pi)
            # create "empty" record:
            self.db_entry = self.init_db_entry()
            # populate with basic info:
            for k in _obs_info:
                self.db_entry[k] = _obs_info[k]
        else:
            self.db_entry = _db_entry

        # print(self.db_entry)
        # pass on the config
        assert _config is not None, 'must pass config to RoboaoObservation ' + _id
        self.config = _config

    def check_db_entry(self):
        """
            Check if DB entry reflects reality
            Might add more checks in the future. Currently only checks pipelining status
        :return:
        """
        try:
            _date = self.db_entry['date_utc'].strftime('%Y%m%d')

            _pipe_names = ['bright_star', 'faint_star']

            for _pipe_name in _pipe_names:

                # pipe self
                _path_pipe = os.path.join(self.config['path']['path_archive'], _date, self.id, _pipe_name)

                # path exists? if yes -- processing must have occurred
                if (_pipe_name in self.db_entry['pipelined']) and os.path.exists(_path_pipe):
                    # check folder modified date:
                    time_tag = datetime.datetime.utcfromtimestamp(os.stat(_path_pipe).st_mtime)
                    # time_tag = mdate_walk(_path_pipe)
                    # bad time tag? force redo!
                    if abs((time_tag - self.db_entry['pipelined'][_pipe_name]['last_modified']).total_seconds()) > 1.0:
                        return {'status': 'ok', 'message': 'DB entry for {:s} does not reflect reality'.format(self.id),
                                'db_record_update': ({'_id': self.id},
                                                     {'$unset': {
                                                         'pipelined.{:s}'.format(_pipe_name): 1
                                                     }}
                                                     )
                                }
                # path does not exist? make sure it's not present in DB entry and/or not marked 'done'
                elif (_pipe_name in self.db_entry['pipelined']) and \
                        self.db_entry['pipelined'][_pipe_name]['status']['done']:
                    return {'status': 'ok', 'message': 'DB entry for {:s} does not reflect reality'.format(self.id),
                            'db_record_update': ({'_id': self.id},
                                                 {'$unset': {
                                                     'pipelined.{:s}'.format(_pipe_name): 1
                                                 }}
                                                 )
                            }

            return {'status': 'ok', 'message': None}

        except Exception as _e:
            traceback.print_exc()
            return {'status': 'error', 'message': str(_e)}

    def get_task(self):
        """
            Figure out what needs to be done with the observation.
            Here is where the processing decision chain is defined
        :return:
        """
        _task = None

        # TODO: BSP?
        ''' Bright star pipeline '''
        pipe = RoboaoBrightStarPipeline(_config=self.config, _db_entry=self.db_entry)
        # check conditions necessary to run (defined in config.json):
        go = pipe.check_necessary_conditions()
        print(go)

        # good to go?
        if go:
            # should and can run BSP pipeline itself?
            _part = 'bright_star_pipeline'
            go = pipe.check_conditions(part=_part)
            if go:
                pipe.db_entry['pipelined']['{:s}'.format(pipe.name)]['status']['enqueued'] = True
                # pipe.db_entry['pipelined']['{:s}'.format(pipe.name)]['last_modified'] = utc_now()
                _task = {'task': _part, 'id': self.id, 'config': self.config, 'db_entry': self.db_entry,
                         'db_record_update': ({'_id': self.id},
                                              {'$set': {
                                                  'pipelined.{:s}'.format(pipe.name):
                                                      pipe.db_entry['pipelined']['{:s}'.format(pipe.name)]
                                              }}
                                              )
                         }
                return _task

            # should and can run Strehl calculation?
            _part = 'bright_star_pipeline:strehl'
            go = pipe.check_conditions(part=_part)
            if go:
                pipe.db_entry['pipelined']['{:s}'.format(pipe.name)]['strehl']['status']['enqueued'] = True
                # pipe.db_entry['pipelined']['{:s}'.format(pipe.name)]['strehl']['last_modified'] = utc_now()
                _task = {'task': _part, 'id': self.id, 'config': self.config, 'db_entry': self.db_entry,
                         'db_record_update': ({'_id': self.id},
                                              {'$set': {
                                                  'pipelined.{:s}'.format(pipe.name):
                                                      pipe.db_entry['pipelined']['{:s}'.format(pipe.name)]
                                              }}
                                              )
                         }
                return _task
            # should and can run PCA pipeline?
            _part = 'bright_star_pipeline:pca'
            go = pipe.check_conditions(part=_part)
            if go:
                pipe.db_entry['pipelined']['{:s}'.format(pipe.name)]['pca']['status']['enqueued'] = True
                # pipe.db_entry['pipelined']['{:s}'.format(pipe.name)]['pca']['last_modified'] = utc_now()
                _task = {'task': _part, 'id': self.id, 'config': self.config, 'db_entry': self.db_entry,
                         'db_record_update': ({'_id': self.id},
                                              {'$set': {
                                                  'pipelined.{:s}'.format(pipe.name):
                                                      pipe.db_entry['pipelined']['{:s}'.format(pipe.name)]
                                              }}
                                              )
                         }
                return _task

        # TODO: FSP?

        # TODO: EOP?

        # FIXME: do something something else?

        return _task

    def check_raws(self, _location, _date, _date_raw_data):
        """
            Check if raw data files info in DB is correct and up-to-date
        :param _location:
        :param _date:
        :param _date_raw_data:
        :return:
        """
        try:
            # raw file names
            _raws = [_s for _s in _date_raw_data if re.match(re.escape(self.id), _s) is not None]
            # deleted?!
            if len(_raws) == 0:
                self.db_entry['raw_data']['location'] = []
                self.db_entry['raw_data']['data'] = []
                self.db_entry['raw_data']['last_modified'] = utc_now()
                return {'status': 'error', 'message': 'raw files for {:s} not available any more'.format(self.id),
                        'db_record_update': ({'_id': self.id},
                                             {'$set': {
                                                 'raw_data.location': [],
                                                 'raw_data.data': [],
                                                 'raw_data.last_modified': self.db_entry['raw_data']['last_modified']
                                             }}
                                             )
                        }
            # time tags. use the 'freshest' time tag for 'last_modified'
            time_tags = [datetime.datetime.utcfromtimestamp(os.stat(os.path.join(_location, _date, _s)).st_mtime)
                         for _s in _raws]
            time_tag = max(time_tags)

            # init/changed? the archiver will have to update database entry then:
            if (len(self.db_entry['raw_data']['data']) == 0) or \
                    (abs((time_tag - self.db_entry['raw_data']['last_modified']).total_seconds()) > 1.0):
                self.db_entry['raw_data']['location'] = ['{:s}:{:s}'.format(
                                                            self.config['server']['analysis_machine_external_host'],
                                                            self.config['server']['analysis_machine_external_port']),
                                                            _location]
                self.db_entry['raw_data']['data'] = sorted(_raws)
                self.db_entry['raw_data']['last_modified'] = time_tag
                # DB updates are handled by the main archiver process
                # we'll provide it with proper query to feed into pymongo's update_one()
                return {'status': 'ok', 'message': 'raw files changed',
                        'db_record_update': ({'_id': self.id},
                                             {'$set': {
                                                    'raw_data.location': self.db_entry['raw_data']['location'],
                                                    'raw_data.data': self.db_entry['raw_data']['data'],
                                                    'raw_data.last_modified': time_tag
                                             }}
                                             )
                        }
            else:
                return {'status': 'ok', 'message': None}

        except Exception as _e:
            traceback.print_exc()
            return {'status': 'error', 'message': str(_e)}

    def check_distributed(self):
        try:
            return {'status': 'ok', 'message': None}

        except Exception as _e:
            traceback.print_exc()
            return {'status': 'error', 'message': _e}

    def parse(self, _program_pi):
        """
            Parse obs info (e.g. contained in id, or fits header) to be injected into DB
            Define decision chain for observation pipelining, i.e. decide what _should_ be done
            without actually checking what's already done

        :param _program_pi: dict program_num -> PI

        :return:
        """
        _obs = self.id
        # parse name:
        _tmp = _obs.split('_')
        # program num. it will be a string in the future
        _prog_num = str(_tmp[0])
        # who's pi?
        if (_program_pi is not None) and (_prog_num in _program_pi.keys()):
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

        # telescope: TODO: move this to config.json
        if _date_utc > datetime.datetime(2015, 10, 1):
            _telescope = 'KPNO_2.1m'
        else:
            _telescope = 'Palomar_P60'

        return {
            'science_program': {
                'program_id': _prog_num,
                'program_PI': _prog_pi
            },
            'name': _sou_name,
            'filter': _filt,
            'date_utc': _date_utc,
            'marker': _marker,
            'camera': _camera,
            'telescope': _telescope
        }

    def init_db_entry(self):
        """
                A dummy database record for a science observation
            :return:
            """
        time_now_utc = utc_now()
        return {
            '_id': self.id,
            'date_added': time_now_utc,
            'name': None,
            'alternative_names': [],
            'science_program': {
                'program_id': None,
                'program_PI': None
            },
            'date_utc': None,
            'telescope': None,
            'camera': None,
            'filter': None,
            'exposure': None,
            'magnitude': None,
            'coordinates': {
                'epoch': None,
                'radec': None,
                'radec_str': None,
                # 'radec_geojson': None,
                'azel': None
            },
            'pipelined': {},

            'seeing': {
                'median': None,
                'mean': None,
                'nearest': None,
                'last_modified': time_now_utc
            },
            'distributed': {
                'status': False,
                'location': [],
                'last_modified': time_now_utc
            },
            'raw_data': {
                'location': [],
                'data': [],
                'last_modified': time_now_utc
            },
            'comment': None
        }


class Pipeline(object):
    def __init__(self, _config, _db_entry):
        """
            Pipeline
        :param _config: dict with configuration
        :param _db_entry: observation DB entry
        """
        self.config = _config
        self.db_entry = _db_entry
        # pipeline name
        # self.name = None
        # this is what gets injected into DB
        # self.status = OrderedDict()

    def check_necessary_conditions(self):
        """
            Check if should be run on an obs
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def init_status():
        """
            Initialize status dict
        :return:
        """
        raise NotImplementedError

    def run(self, part):
        """
            Run the pipeline
            # :param aux: auxiliary data including calibration
        :return:
        """
        raise NotImplementedError


class RoboaoBrightStarPipeline(Pipeline):
    """
        Robo-AO's Bright Star Pipeline
    """
    def __init__(self, _config, _db_entry):
        """
            Init Robo-AO's Bright Star Pipeline
            :param _config: dict with configuration
            :param _db_entry: observation DB entry
        """
        ''' initialize super class '''
        super(RoboaoBrightStarPipeline, self).__init__(_config=_config, _db_entry=_db_entry)

        # pipeline name. This goes to 'pipelined' field of obs DB entry
        self.name = 'bright_star'

        # initialize status
        if self.name not in self.db_entry['pipelined']:
            self.db_entry['pipelined'][self.name] = self.init_status()

    def check_necessary_conditions(self):
        """
            Check if should be run on an obs (if necessary conditions are met)
            :param _db_entry: observation DB entry
        :return:
        """
        go = True

        for field_name, field_condition in self.config['pipeline'][self.name]['go'].items():
            if field_name != 'help':
                # build proper dict reference expression
                keys = field_name.split('.')
                expr = 'self.db_entry'
                for key in keys:
                    expr += "['{:s}']".format(key)
                # get condition
                condition = eval(expr + ' ' + field_condition)
                # eval condition
                go = go and condition

        return go

    def check_conditions(self, part=None):
        """
            Perform condition checks for running specific parts of pipeline
            :param part: which part of pipeline to run
        :return:
        """
        assert part is not None, 'must specify what to check'

        # check the BSP itself?
        if part == 'bright_star_pipeline':
            # force redo requested?
            _force_redo = self.db_entry['pipelined'][self.name]['status']['force_redo']
            # pipeline done?
            _done = self.db_entry['pipelined'][self.name]['status']['done']
            # how many times tried?
            _num_tries = self.db_entry['pipelined'][self.name]['status']['retries']

            go = _force_redo or ((not _done) and (_num_tries <= self.config['misc']['max_retries']))

            return go

        # Strehl calculation for the results of BSP processing?
        elif part == 'bright_star_pipeline:strehl':

            return False

        # Run PCA high-contrast processing pipeline?
        elif part == 'bright_star_pipeline:pca':

            return False

    @staticmethod
    def init_status():
        time_now_utc = utc_now()
        return {
            'status': {
                'done': False,
                'enqueued': False,
                'force_redo': False,
                'retries': 0,
            },
            'last_modified': time_now_utc,
            'preview': {
                'done': False,
                'force_redo': False,
                'retries': 0,
                'last_modified': time_now_utc
            },
            'location': [],
            'classified_as': None,
            'fits_header': {},
            'strehl': {
                'status': {
                    'force_redo': False,
                    'enqueued': False,
                    'done': False,
                    'retries': 0
                },
                'lock_position': None,
                'ratio_percent': None,
                'core_arcsec': None,
                'halo_arcsec': None,
                'fwhm_arcsec': None,
                'flag': None,
                'last_modified': time_now_utc
            },
            'pca': {
                'status': {
                    'force_redo': False,
                    'enqueued': False,
                    'done': False,
                    'retries': 0
                },
                'preview': {
                    'force_redo': False,
                    'done': False,
                    'retries': 0
                },
                'location': [],
                'lock_position': None,
                'contrast_curve': None,
                'last_modified': time_now_utc
            }
        }

    def generate_lucky_settings_file(self, out_settings, _path_calib,
                                     all_files, final_gs_x, final_gs_y, gs_diam,
                                     final_bg_x, final_bg_y, bg_diam):
        with open(out_settings, 'w') as f_out_settings:
            settings_out_n = 0
            for l in open(self.config['pipeline']['bright_star']['pipeline_settings_template'], 'r'):
                if l.find("???") == 0:
                    if settings_out_n == 0:
                        file_n = 0
                        for f in all_files:
                            with fits.open(f) as _tmp:
                                n_frames = len(_tmp)
                            f_out_settings.write('{:<7d}{:s}    0-{:d}\n'.format(file_n, f, n_frames - 1))
                            file_n += 1
                        settings_out_n += 1
                    elif settings_out_n == 1:
                        for file_n, f in enumerate(all_files):
                            f_out_settings.write('{:<8d}{:<8d}({:d},{:d}),({:d},{:d})\n'
                                                 .format(file_n, 1,
                                                         final_gs_x - gs_diam // 2, final_gs_y - gs_diam // 2,
                                                         final_gs_x + gs_diam // 2, final_gs_y + gs_diam // 2))
                        settings_out_n += 1
                    elif settings_out_n == 2:
                        for file_n, f in enumerate(all_files):
                            f_out_settings.write('{:<8d}({:d},{:d}),({:d},{:d})\n'
                                                 .format(file_n,
                                                         final_bg_x - bg_diam // 2, final_bg_y - bg_diam // 2,
                                                         final_bg_x + bg_diam // 2, final_bg_y + bg_diam // 2))
                        settings_out_n += 1
                    elif settings_out_n == 3:
                        with fits.open(all_files[0]) as f_fits:
                            camera_mode = f_fits[0].header['MODE_NUM']
                            naxis1 = int(f_fits[0].header['NAXIS1'])
                            if naxis1 == 256:
                                camera_mode = repr(camera_mode) + '4'
                            else:
                                camera_mode = repr(camera_mode)
                        f_out_settings.write('Bias file (filename/none)                  : ' +
                                             os.path.join(_path_calib, 'dark_{:s}.fits\n'.format(camera_mode)))
                        settings_out_n += 1
                    elif settings_out_n == 4:
                        f_out_settings.write('Flat (filename/none)                       : ' +
                                             os.path.join(_path_calib,
                                                          'flat_{:s}.fits\n'.format(self.db_entry['filter'])))
                        settings_out_n += 1

                else:
                    f_out_settings.write(l)

    def run(self, part=None):
        """
            Execute specific part of pipeline
        :return:
        """
        # TODO:
        assert part is not None, 'must specify part to execute'

        # UTC date of obs:
        _date = self.db_entry['date_utc'].strftime('%Y%m%d')

        # path to store unzipped raw files
        _path_tmp = self.config['path']['path_tmp']
        # path to raw files:
        _path_raw = os.path.join(self.db_entry['raw_data']['location'][1], _date)
        # path to lucky-pipelined data:
        _path_archive = os.path.join(self.config['path']['path_archive'], _date)
        # path to calibration data produced by lucky pipeline:
        _path_calib = os.path.join(self.config['path']['path_archive'], _date, 'calib')

        if part == 'bright_star_pipeline':
            # steps from reduce_data_multithread.py + image_reconstruction.cpp

            # raw files:
            _raws_zipped = sorted(self.db_entry['raw_data']['data'])

            # unbzip raw source file(s):
            lbunzip2(_path_in=_path_raw, _files=_raws_zipped, _path_out=_path_tmp, _keep=True)

            # unzipped file names:
            raws = [os.path.join(_path_tmp, os.path.splitext(_f)[0]) for _f in _raws_zipped]

            ''' go off with processing '''
            bg_diam = 50
            gs_diam = 50

            base_val = 0

            files_to_analyse = []
            for f in sorted(raws):
                # try:
                if (len(f.split(".")) < 2 and f.split(".")[0] != '_') or f.split(".")[-2][-2] != '_':
                    orig_f = f
                    with fits.open(f) as p:
                        # sometimes the first file output is actually the smallest in number of frames
                        if len(p) < 50:
                            new_fn = f.rsplit(".", 1)[-2] + '_0.fits'
                            if os.path.exists(new_fn):
                                f = new_fn
                        # fs = f.split("/")[-1]
                        files_to_analyse.append([f, orig_f])
                # except Exception as _e:
                #     print(_e)
                #     print("Failed to process", f)

            for fn, orig_fn in sorted(files_to_analyse):
                # print('_____________________')
                # print(files_to_analyse)
                print("Analysing", orig_fn)
                # make a directory to store this run
                out_dir = os.path.join(_path_archive, self.db_entry['_id'], self.name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                # first make a preview sum image
                with fits.open(fn) as p:
                    img_size = p[0].data.shape

                    # make 5 frames and median combine them to avoid selecting cosmic rays as the guide star
                    print("Getting initial frame average")
                    avg_imgs = np.zeros((5, img_size[0], img_size[1]))

                    for avg_n in range(0, 5):
                        n_avg_frames = 0.0
                        for frame_n in list(range(avg_n, len(p), 5))[::4]:
                            avg_imgs[avg_n] += p[frame_n].data + base_val
                            n_avg_frames += 1.0
                        avg_imgs[avg_n] /= n_avg_frames

                    avg_img = np.median(avg_imgs, axis=0)
                    export_fits(os.path.join(out_dir, 'sum.fits'), avg_img)

                    mid_portion = avg_img[30:avg_img.shape[0] - 30, 30:avg_img.shape[1] - 30]

                    # if there's a NaN something's gone horribly wrong
                    if np.sum(mid_portion) != np.sum(mid_portion):
                        classified_as = 'failed'
                        self.db_entry['pipelined'][self.name]['classified_as'] = classified_as
                        raise RuntimeError('Something went horribly wrong')

                    print(mid_portion.shape)

                    mid_portion = ndimage.gaussian_filter(mid_portion, sigma=10)

                    # subtract off a much more smoothed version to remove large-scale gradients across the image
                    mid_portion -= ndimage.gaussian_filter(mid_portion, sigma=60)

                    mid_portion = mid_portion[30:mid_portion.shape[0] - 30, 30:mid_portion.shape[1] - 30]

                    # pyfits.PrimaryHDU(mid_portion).writeto("filtered_img.fits",clobber=True)

                    final_gs_y, final_gs_x = np.unravel_index(mid_portion.argmax(), mid_portion.shape)
                    final_gs_y += 60
                    final_gs_x += 60
                    print("\tGuide star selected at:", final_gs_x, final_gs_y)

                    # now guess the background region
                    final_bg_x = final_gs_x + gs_diam
                    final_bg_y = final_gs_y + gs_diam

                    if final_bg_x > mid_portion.shape[1]:
                        final_bg_x = mid_portion.shape[1]
                    if final_bg_y > mid_portion.shape[0]:
                        final_bg_y = mid_portion.shape[0]

                    print("\tGenerating S&A image")

                    # now do a really simple S&A preview, using the selected guide star
                    # also track the SNR of the star to get an estimate for the
                    # imaging performance
                    with open(os.path.join(out_dir, 'quicklook_stats.txt'), 'w') as stats_out:

                        snrs = []
                        output_image = np.zeros((img_size[0] * 2, img_size[1] * 2))
                        for frame_n in range(0, len(p), 3):
                            if frame_n % 9 == 0:
                                # print("\r\t", frame_n, "/", len(p))
                                # sys.stdout.flush()
                                pass

                            gs_region = p[frame_n].data[final_gs_y - gs_diam // 2: final_gs_y + gs_diam // 2,
                                        final_gs_x - gs_diam // 2: final_gs_x + gs_diam // 2] + base_val
                            gs_region = gs_region.astype(float)

                            # using the BG region as an estimate of the background noise
                            bg_region = p[frame_n].data[final_bg_y - gs_diam // 4: final_bg_y + gs_diam // 4,
                                        final_bg_x - gs_diam // 4:final_bg_x + gs_diam // 4]
                            bg_rms = np.std(bg_region)
                            bg_base = np.average(bg_region)

                            blurred_version = ndimage.gaussian_filter(gs_region, sigma=1)

                            max_posn = np.unravel_index(blurred_version.argmax(), blurred_version.shape)
                            max_posn_y = max_posn[0] + final_gs_y - gs_diam // 2
                            max_posn_x = max_posn[1] + final_gs_x - gs_diam // 2

                            # region around guide star, to include PSF-convolution
                            signal = np.sum(
                                gs_region[max_posn[0] - 2: max_posn[0] + 3, max_posn[1] - 2: max_posn[1] + 3] - bg_base)

                            snrs.append(signal / bg_rms)

                            stats_out.write('{:d} {:f} {:f} {:f} {:f} {:f}\n'.format(frame_n, signal / bg_rms,
                                                                                     signal, bg_rms, bg_base,
                                                    np.sum(gs_region[max_posn[0] - 2 - 20:max_posn[0] + 3 - 20,
                                                           max_posn[1] - 2 - 20:max_posn[1] + 3 - 20] - bg_base)))

                            dx = final_gs_x - max_posn_x
                            dy = final_gs_y - max_posn_y

                            out_start_x = img_size[1] // 2 + dx
                            out_start_y = img_size[0] // 2 + dy

                            if (0 < out_start_x < img_size[1]) and (0 < out_start_y < img_size[0]):
                                output_image[out_start_y:out_start_y + img_size[0],
                                             out_start_x:out_start_x + img_size[1]] += p[frame_n].data + base_val
                        output_image = output_image[20 + img_size[1] // 2: (img_size[1] * 3) // 2 - 20,
                                       20 + img_size[0] // 2: (img_size[0] * 3) // 2 - 20]
                        export_fits(os.path.join(out_dir, self.db_entry['_id'] + '_preview_saa.fits'), output_image)

                    valid_snrs = []
                    for s in snrs:
                        if 0.0 < s < 1.0e8:
                            valid_snrs.append(s)
                    valid_snrs = sigma_clip(valid_snrs, 3, 2)
                    snr = np.average(valid_snrs)
                    print("\tSNR: {:.1f}".format(snr))

                    print("\tGenerating lucky settings file")
                    # first find all the files for this target
                    extra_files = glob.glob(fn.rsplit(".", 1)[0] + "_?.fits")
                    all_files = [fn]
                    for e in extra_files:
                        all_files.append(e)

                    self.generate_lucky_settings_file(os.path.join(out_dir, 'pipeline_settings.txt'), _path_calib,
                                                      all_files, final_gs_x, final_gs_y, gs_diam,
                                                      final_bg_x, final_bg_y, bg_diam)

                    if snr < 20.0 or len(snrs) < 20:
                        classified_as = 'zero_flux'
                    elif snr < 100.0:
                        classified_as = 'faint'
                    else:
                        classified_as = 'high_flux'
                    self.db_entry['pipelined'][self.name]['classified_as'] = classified_as

                    # run c++ code
                    subprocess.run(['{:s} pipeline_settings.txt'.format(
                        self.config['pipeline'][self.name]['pipeline_executable'])],
                        check=True, shell=True, cwd=out_dir)

                    # reduction successful? prepare db entry for update
                    f100p = os.path.join(out_dir, '100p.fits')
                    if os.path.exists(f100p):
                        self.db_entry['pipelined'][self.name]['status']['done'] = True
                        # save fits header
                        self.db_entry['pipelined'][self.name]['fits_header'] = get_fits_header(f100p)
                    else:
                        self.db_entry['pipelined'][self.name]['status']['done'] = False
                    self.db_entry['pipelined'][self.name]['status']['enqueued'] = False
                    self.db_entry['pipelined'][self.name]['status']['force_redo'] = False
                    self.db_entry['pipelined'][self.name]['status']['retries'] += 1

                    # set last_modified as out_dir folder modified date:
                    time_tag = datetime.datetime.utcfromtimestamp(os.stat(out_dir).st_mtime)
                    self.db_entry['pipelined'][self.name]['last_modified'] = time_tag

                for _file in raws:
                    print('removing', _file)
                    os.remove(os.path.join(_file))

        elif part == 'bright_star_pipeline:strehl':
            # compute strehl

            return {'status': 'ok', 'message': None}

        elif part == 'bright_star_pipeline:pca':
            # run high-contrast pipeline

            return {'status': 'ok', 'message': None}

        else:
            return {'status': 'error', 'message': 'unknown pipeline part'}


def job_bright_star_pipeline(_id=None, _config=None, _db_entry=None, _task_hash=None):
    try:
        # init pipe here again. [as it's not JSON serializable]
        pip = RoboaoBrightStarPipeline(_config=_config, _db_entry=_db_entry)
        # TODO: run the pipeline
        pip.run(part='bright_star_pipeline')

        return {'_id': _id, 'job': 'bright_star_pipeline', 'hash': _task_hash,
                'status': 'ok', 'message': str(datetime.datetime.now()),
                'db_record_update': ({'_id': _id},
                                     {'$set': {
                                         'pipelined.bright_star': pip.db_entry['pipelined']['bright_star']
                                     }}
                                     )
                }
    except Exception as _e:
        traceback.print_exc()
        try:
            _status = _db_entry['pipelined']['bright_star']
        except Exception as _ee:
            print(str(_ee))
            traceback.print_exc()
            # failed? flush status:
            _status = RoboaoBrightStarPipeline.init_status()
        # retries++
        _status['status']['retries'] += 1
        _status['status']['enqueued'] = False
        _status['status']['force_redo'] = False
        _status['last_modified'] = utc_now()
        return {'_id': _id, 'job': 'bright_star_pipeline', 'hash': _task_hash,
                'status': 'error', 'message': str(_e),
                'db_record_update': ({'_id': _id},
                                     {'$set': {
                                         'pipelined.bright_star': _status
                                     }}
                                     )
                }


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
