import signal
from distributed import Client, LocalCluster
from queue import Queue
import time
import datetime
import inspect
import os
import traceback
import sys
import json
import logging
import collections


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
                if old_time_left > 0: # deduct f's run time from the saved timer
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
            self.logger, self.logger_utc_date = self.set_up_logging(_name='archive', _level=logging.INFO, _mode='a')

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
            cluster = LocalCluster(n_workers=self.config['parallel']['n_workers'],
                                   threads_per_worker=self.config['parallel']['threads_per_worker'])
            self.c = Client(cluster)

            ''' set up processing queue '''
            self.q = OrderedSetQueue()

            # now we need to map our queue over task_runner and gather results in another queue.
            # user must setup specific tasks/jobs in task_runner, which (unfortunatelly)
            # cannot be defined inside a subclass -- only as a standalone function
            self.futures = self.c.map(task_runner, self.q, maxsize=self.config['parallel']['n_workers'])
            self.results = self.c.gather(self.futures)  # Gather results

        except Exception as e:
            print(e)
            traceback.print_exc()
            sys.exit()

    @staticmethod
    def get_config(_config_file):
        """
            Load config JSON file
        """
        # raise NotImplementedError
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
            # config should not be empty:
            if len(config_data) > 0:
                return config_data
            else:
                raise Exception('Failed to load config file')

    def set_up_logging(self, _name='archive', _level=logging.DEBUG, _mode='w'):
        """ Set up logging

            :param _name:
            :param _level: DEBUG, INFO, etc.
            :param _mode: overwrite log-file or append: w or a
            :return: logger instance
            """

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
            self.logger, self.logger_utc_date = self.set_up_logging(_name='archive', _level=logging.INFO, _mode='a')


class RoboaoArchiver(Archiver):
    def __init__(self, config_file=None):
        # initialize super class
        super(RoboaoArchiver, self).__init__(config_file=config_file)


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


if __name__ == '__main__':

    arch = RoboaoArchiver('config.json')

    # fake tasks:
    tasks = [json.dumps({'task': 'test', 'a': aa}) for aa in range(20)]
    tasks += [json.dumps({'task': 'bogus', 'a': aa}) for aa in range(20, 50)]

    for task in tasks:
        arch.q.put(task)

    time.sleep(30)