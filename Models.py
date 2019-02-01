import sqlalchemy as db
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.expression import func
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey
import datetime
import dill
import json
import numpy as np

Base = declarative_base()

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
    
class Session(Base):
    __tablename__ = 'sessions'

    id = Column(Integer, primary_key=True)
    children = relationship("Session", backref=backref('parent', remote_side=[id]))
    parent_id = Column(Integer, ForeignKey('sessions.id'))
    iteration = Column(Integer)

    agent_class = Column(String)
    agent_config = Column(String)
    
    env_name = Column(String)
    env_config = Column(String)
    
    weights = Column(String)
    commit = Column(String)

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
        
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)