import numpy as np
import json
import logging
import os
import shutil

class MarkovDecisionProcess:
    def __init__(self, name='mdp', version=0, path='mdps/',
                 save=True, verbose=False, overwrite=False, *args, **kargs):
        self.name = name
        self.version = version
        if path[-1] != '/':
            path.append('/')
        self.path = path
        self.save = save
        self.verbose = verbose
        self.overwrite = overwrite
        self.kargs = kargs
        
        self.handlers = {}
        self.loggers = {}
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.mdp_path = "{}{}-{}/".format(self.path, self.name, self.version)
        self.model_path = self.mdp_path + 'models/'
        self.data_path = self.mdp_path + 'data/'
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def increment_version(self):
        extra_version = 0
        while True:
            self.version = len(self.get_previous_run_versions()) + extra_version
            proposed_mdp_path = "{}{}-{}/".format(self.path, self.name, self.version)
            if os.path.isdir(proposed_mdp_path):
                extra_version += 1
            else:
                break
            if extra_version > 999:
                raise StopIteration('Exceeded 999 runs of the name {}'.format(self.name))
    
    def create_log_dir(self):
        if os.path.isdir(self.path):
            if self.overwrite:
                shutil.rmtree(self.mdp_path)
            else:
                raise FileExistsError("{} exists and overwrite=False.".format(self.mdp_path))
                
        # Make sure our paths exist
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.mdp_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
    def get_previous_run_versions(self):
        try:
            files = os.listdir(self.path)
            files = [f for f in files if self.name + '-' in f]
            return files
        except FileNotFoundError as e:
            print("Root path {} doesn't exist.  Creating it...".format(self.path))
            os.makedirs(self.path, exist_ok=True)
            return []
        
    def setup_logger(self, name, sub_path=''):
        name = sub_path + name
        logger_path = '{}{}{}.txt'.format(self.data_path, sub_path, name)
        handler = logging.FileHandler(logger_path)
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        
        if self.verbose:
            print('verbose')
            console_handler = logging.StreamHandler()
            logger.addHandler(console_handler)
            
        self.handlers[name] = handler
        self.loggers[name] = logger
        
        return self.loggers[name]
    
    def log(self, value, name, sub_path=''):
        name = sub_path + name
        self.loggers[name].info(NumpyMessage(value))
        
    def get_log(self, name, sub_path=''):
        name = sub_path + name
        self.close_log(name)
        logger_path = '{}{}{}.txt'.format(self.data_path, sub_path, name)
        data = []
        with open(logger_path) as f:
            for line in f.readlines():
                data.append(json.loads(line))
                
        return data
    
    def close_log(self, name=None):
        if name is None:
            names = list(self.loggers.keys())[:]
            for n in names:
                self.loggers.pop(n)
                self.handlers.pop(n).close()
        else:
            if name in self.loggers.keys():
                self.loggers.pop(name)
                self.handlers.pop(name).close()
                
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class NumpyMessage(object):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return json.dumps(self.data, cls=NumpyEncoder)