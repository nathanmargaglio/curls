import datetime
import dill
import json
import sys
import os
import argparse

import numpy as np
from dotenv import load_dotenv
from git import Repo

import sqlalchemy as db
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.expression import func
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey, desc

Base = declarative_base()

def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class SessionManager:
    def __init__(self, *args, **kargs):
        load_dotenv()
        self.database_url = os.getenv('DATABASE_URL')
        self.engine = db.create_engine(self.database_url)
        self.connection = self.engine.connect()
        self.DBSession = sessionmaker(bind=self.engine)
        self.db = self.DBSession()
        self.configured = False
        
        self.SessionClass = Session
        self.EpisodeClass = Episode
        self.StepClass = Step
        Base.metadata.create_all(self.engine)
        self.configure()
        
    def configure(self, arg_list=[]):
        parser = argparse.ArgumentParser(
            description='Session Manager.')
        
        parser.add_argument('-p', '--parent', nargs='?', default=None, type=int,
                           help="Which parent Session to branch from.")
        parser.add_argument('-r', '--rule', nargs='?', default='max-reward', type=str,
                           help="Which Session to branch from upon finishing.")
        parser.add_argument('-a', '--agent-id', nargs='?', default=None, type=int,
                           help="Agent ID to branch from (default to parent Session's)")
        parser.add_argument('-e', '--env-name', nargs='?', default='CartPole-v1', type=str,
                           help="Gym Environment name to load.")
        
        parser.parse_args(arg_list, namespace=self)
        self.configured = True
        
    def _get_parent_from_rule(self):
        if self.rule == 'max-reward':
            print(f"_get_parent_from_rule: self.rule={self.rule}")
            return self.db.query(Session).order_by(desc(Session.average_reward)).limit(1).first()
        else:
            raise NameError(f"Rule {self.rule} is not defined.")
        
    def initialize_session(self):
        if self.parent is not None:
            parent = self.db.query(Session).get(self.parent)
        else:
            parent = self._get_parent_from_rule()
            
        if parent is None:
            iteration = 0
        else:
            iteration = parent.iteration + 1
            
        if self.agent_id is not None:
            agent = self.db.query(Agent).get(self.agent_id)
        elif parent is None:
            agent = Agent()
        else:
            agent = parent.agent
            
        try:
            repo = Repo('./')
            git_url = [url for url in repo.remote().urls][0]
            commit = repo.head.object.hexsha
        except:
            git_url = None
            commit = None
            
        self.session = Session(parent=parent, agent=agent, env_name=self.env_name,
                               iteration=iteration, commit=commit, git_url=git_url)
        self.db.add(self.session)
        self.db.commit()

class Agent(Base):
    __tablename__ = 'agents'
    id = Column(Integer, primary_key=True)
    agent_class = Column(String)
    config = Column(String)
    weights = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"Agent {self.id}"
    
class Session(Base):
    __tablename__ = 'sessions'

    id = Column(Integer, primary_key=True)
    children = relationship("Session", backref=backref('parent', remote_side=[id]))
    parent_id = Column(Integer, ForeignKey('sessions.id'))
    
    agent_id = Column(Integer, ForeignKey('agents.id'))
    agent = relationship("Agent", backref="agent")
    
    iteration = Column(Integer)
    env_name = Column(String)
    commit = Column(String)
    git_url = Column(String)
    average_reward = Column(Float)

    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    elapse_time = Column(Float)

    def __repr__(self):
        return f"Session {self.id}: {self.iteration}"

    def __str__(self):
        s =  f"id      : {self.id}\n"
        s += f"iter.   : {self.iteration}\n"
        s += f"parent  : {self.parent_id}\n"
        s += f"children: {self.children}\n"
        s += f"env_name: {self.env_name}\n"
        s += f"commit  : {self.commit}\n"
        s += f"start_t : {self.start_time}\n"
        s += f"elapse_t: {self.elapse_time}\n"
        return s
    
    def set_data(self, data):
        self.iteration = data["iteration"]
        self.parent_id = data["parent_id"] if "parent_id" in data else None
        self.agent_class = dill.dumps(data["agent_class"]) if "agent_class" in data else None
        self.agent_config = json.dumps(data["agent_config"], cls=NumpyEncoder) if "agent_config" in data else None
        self.env_name = data["env_name"] if "env_name" in data else None
        self.env_config = json.dumps(data["env_config"], cls=NumpyEncoder) if "env_config" in data else None
        self.weights = json.dumps(data["eights"], cls=NumpyEncoder)
        self.commit = data["commit"] if "commit" in data else None
        
    def __call__(self):
        return {
            "id": self.id,
            "iteration": self.iteration,
            "parent_id": self.parent_id,
            "agent_class": dill.loads(self.agent_class),
            "agent_config": json.loads(self.agent_config),
            "env_name": self.env_name,
            "env_config": json.loads(self.env_config),
            "weights": json.loads(self.weights),
            "commit": self.commit,
            "start_time": str(self.start_time),
            "elapse_time": self.elapse_time
        }

class Episode(Base):
    __tablename__ = 'episodes'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'))
    session = relationship("Session", backref="episodes")
    iteration = Column(Integer)
    total_reward = Column(Float)
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    elapse_time = Column(Float)

    def __repr__(self):
        return f"Episode {self.id}: {self.iteration}"

    def __str__(self):
        return f"Episode {self.id}"
    
    def __str__(self):
        s =  f"id      : {self.id}\n"
        s += f"iter.   : {self.iteration}\n"
        s += f"session : {self.session_id}\n"
        s += f"reward  : {self.total_reward}\n"
        s += f"start_t : {self.start_time}\n"
        s += f"elapse_t: {self.elapse_time}\n"
        return s
    
    def set_data(self, data):
        self.iteration = data["iteration"]
        self.session_id = data["session_id"]
        self.total_reward = data["total_reward"]
        
    def __call__(self):
        return {
            "id": self.id,
            "iteration": self.iteration,
            "session_id": self.session_id,
            "total_reward": self.total_reward,
            "start_time": str(self.start_time),
            "elapse_time": self.elapse_time
        }
    
class Step(Base):
    __tablename__ = 'steps'

    id = Column(Integer, primary_key=True)
    iteration = Column(Integer)
    
    episode_id = Column(Integer, ForeignKey('episodes.id'))
    episode = relationship("Episode", backref="steps")
    
    observation = Column(String)
    action = Column(String)
    reward = Column(Float)
    done = Column(Boolean)
    info = Column(String)
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    elapse_time = Column(Float)

    def __repr__(self):
        return f"Step {self.id}: {self.iteration}"
    
    def __str__(self):
        s =  f"id      : {self.id}\n"
        s += f"iter.   : {self.iteration}\n"
        s += f"episode : {self.episode_id}\n"
        s += f"obs.    : {self.observation}\n"
        s += f"action  : {self.action}\n"
        s += f"reward  : {self.reward}\n"
        s += f"done    : {self.done}\n"
        s += f"info    : {self.info}\n"
        s += f"start_t : {self.start_time}\n"
        s += f"elapse_t: {self.elapse_time}\n"
        return s
    
    def set_data(self, data):
        self.iteration = data["iteration"]
        self.episode_id = data["episode_id"]
        self.observation = json.dumps(data["observation"], cls=NumpyEncoder)
        self.action = json.dumps(data["action"], cls=NumpyEncoder)
        self.reward = data["reward"]
        self.info = json.dumps(data["info"], cls=NumpyEncoder)

    def __call__(self):
        return {
            "id": self.id,
            "iteration": self.iteration,
            "episode_id": self.episode_id,
            "observation": json.loads(self.observation),
            "action": json.loads(self.action),
            "reward": self.reward,
            "done": self.done,
            "info": json.loads(self.info),
            "start_time": str(self.start_time),
            "elpase_time": self.elapse_time
        }
        
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)