import zmq
import time
import json
import logging
import os
import shutil
import dill
import datetime
import multiprocessing as mp

import sqlalchemy as db
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.expression import func
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey
Base = declarative_base()

from git import Repo
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

from Models import Step, Episode
    
class MarkovDecisionProcess:
    def __init__(self, name='session', version=0, path='sessions/', memory=False, *args, **kargs):
        self.name = name
        self.version = version
        self.path = path
        self.memory = memory
        if self.memory:
            self.engine = db.create_engine('sqlite://')
        else:
            os.makedirs(path, exist_ok=True)
            self.engine = db.create_engine(f'sqlite:///{path}{name}_{version}.db')

        self.connection = self.engine.connect()
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        self.generate_tables()
        self.Episode = Episode
        self.Step = Step
        
        self.last_called = None
        
    def run(self, agent, env, episode=None, verbose=False, *args, **kargs):
        obs = env.reset()
        done = False
        rewards = []
        actions = []
        observations = []
        dones = []
        infos = []

        step_count = 0
        total_rewards = 0
        while not done:
            logits, _ = agent.model(tf.convert_to_tensor(obs[None, :], dtype=tf.float32))
            probs = tf.nn.softmax(logits)
            action = np.random.choice(env.action_space.n, p=probs.numpy()[0])

            next_obs, reward, done, info = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
            total_rewards += reward
            if episode is not None:
                self.save_step(step_count, episode, obs, action, reward, done, info)
                episode.total_rewards = total_rewards
                self.session.commit()

            obs = next_obs
            step_count += 1
            if done:
                return observations, actions, rewards, dones, infos
            
    def train(self, agent, env, episodes, verbose=False):
        batch = {
            "observations": [],
            "actions": [],
            "discounted_rewards": []
        }
        last_episode = self.session.query(Episode).order_by(Episode.id.desc()).first()
        if last_episode is None:
            start_episode = 0
        else:
            start_episode = last_episode.episode_count + 1
        
        for episode in range(start_episode, start_episode + episodes):
            episode_instance = self.save_episode(episode, 0, agent.__class__, agent.model.get_weights())
            observations, actions, rewards, dones, infos = self.run(agent, env, episode_instance, verbose=verbose)

            batch['observations'] += observations
            batch['actions'] += actions
            batch['discounted_rewards'] += get_discounted_rewards(rewards, agent.gamma)

            if len(batch['observations']) >= agent.batch_size:
                for epoch in range(agent.epochs):
                    agent.learn(**batch)
                batch = {
                    "observations": [],
                    "actions": [],
                    "discounted_rewards": []
                }
            
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.rollback()
        self.session.close()
    
    def rollback(self):
        return self.session.rollback()
    
    def generate_tables(self):
        Base.metadata.create_all(self.engine)
    
    def save_step(self, step_count, episode, observation, action, reward, done, info):
        s = Step(step_count=int(step_count),
                       episode=episode,
                       observation=json.dumps(observation, cls=NumpyEncoder),
                       action=json.dumps(action, cls=NumpyEncoder),
                       reward=float(reward),
                       done=bool(done),
                       info=json.dumps(info, cls=NumpyEncoder)
                      )
        self.session.add(s)
        self.session.commit()
        return s
    
    def save_episode(self, episode, total_reward, agent_class, weights):
        try:
            repo = Repo('./')
            commit = repo.head.object.hexsha
        except:
            commit = None
        
        e = Episode(count=int(episode),
                    total_reward=float(total_reward),
                    agent_class=dill.dumps(agent_class),
                    weights=json.dumps(weights, cls=NumpyEncoder),
                    commit=commit
                   )
        self.session.add(e)
        self.session.commit()
        return e
    
    def tail(self, limit=10):
        return self.session.query(self.Table).order_by(self.Table.id.desc()).limit(limit).all()[::-1]
    
    def last(self, update_last=True):
        last_called = self.session.query(self.Table).order_by(self.Table.id.desc()).first()
        if update_last:
            self.last_called = last_called
        return last_called
    
    def get_episode(self, episode=None):
        if episode is None:
            episode = self.session.query(func.max(self.Table.episode)).first()[0]
        return self.session.query(self.Table).filter(self.Table.episode == episode).all()
    
    def since_last(self, limit=10, update_last=True):
        since_last_query = self.session.query(self.table).order_by(self.table.id.desc())
        if self.last_called is not None:
            since_last_query = since_last_query.filter(table.id > self.last_called.id)
            
        since_last = since_last_query.limit(limit).all()[::-1]
        if update_last and len(since_last):
            self.last_called = since_last[-1]
        return since_last
            
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def get_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    reward_sum = 0
    for reward in rewards[::-1]:
        reward_sum = reward + gamma * reward_sum
        discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()
    return discounted_rewards
    
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