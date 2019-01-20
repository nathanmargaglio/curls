import zmq
import time
import json
import logging
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

def start_master():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)

    header = "master >"
    context = zmq.Context()
    model = Classifier(2, 1)
    losses = []

    def get_loss(model):
        tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        output = model(tensor)
        loss = -tf.reduce_mean((output_data - output)**2)
        return np.abs(loss.numpy())
        
    def apply_grads(model, grads):
        for w, g in zip(model.trainable_weights, grads):
            w.assign_add(g)

    with context.socket(zmq.REP) as socket:
        socket.bind('tcp://127.0.0.1:5555')
        print(header, 'listening...')
        try:
            count = 0
            while True:
                msg = json.loads(socket.recv())
                socket.send_string(json.dumps({ "weights": model.get_weights()}, cls=NumpyEncoder))
                apply_grads(model, msg['grads'])
                loss = get_loss(model)
                losses.append(loss)
                count += 1
                if count % 100 == 0:
                    print("{} {:03d} {}".format(header, count, loss))
        except KeyboardInterrupt as e:
            print(header, 'Stopped.')
            plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(np.abs(losses))
            plt.show()
            
def start_worker(input_data, output_data):
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)

    header = "- {} >".format(input_data)
    context = zmq.Context()
    
    model = Classifier(2, 1)
    losses = []
    lr = 0.01
    input_data = input_data
    output_data = output_data
    tensor = tf.convert_to_tensor(np.array([input_data]), dtype=tf.float32)
    
    def get_grads(model, tensor):
        with tf.GradientTape() as tape:
            output = model(tensor)
            loss = -tf.reduce_mean((output_data - output)**2)
        losses.append(loss.numpy())
        grads = tape.gradient(loss, model.trainable_weights)
        return [lr*g.numpy() for g in grads]
    
    try:
        for i in range(100):
            with context.socket(zmq.REQ) as socket:
                socket.connect('tcp://127.0.0.1:5555')
                grads = get_grads(model, tensor)
                socket.send_string(json.dumps({ "grads": grads, "data": input_data }, cls=NumpyEncoder))
                msg = json.loads(socket.recv())
                model.set_weights(msg['weights'])
    except KeyboardInterrupt as e:
        print(header, 'Stopped.')

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