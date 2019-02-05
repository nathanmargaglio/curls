import datetime
import dill
import json
import sys
import os
import argparse

import gym
from gym import spaces
import numpy as np
from dotenv import load_dotenv
from git import Repo
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

import sqlalchemy as db
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.expression import func
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, LargeBinary, JSON, ForeignKey, desc

Base = declarative_base()

from Agent import ActorCriticAgent

class SessionManager:
    def __init__(self, *args, **kargs):
        load_dotenv()
        self.database_url = os.getenv('DATABASE_URL')
        self.engine = db.create_engine(self.database_url, pool_pre_ping=True)
        self.connection = self.engine.connect()
        self.configured = False
        
        self.SessionClass = Session
        self.EpisodeClass = Episode
        self.AgentClass = Agent
        
    def connect_to_database(self):
        self.DBSession = sessionmaker(bind=self.engine)
        self.db = self.DBSession()
        Base.metadata.create_all(self.engine)
        
    def disconnect_from_database(self):
        self.db.commit()
        self.db.close()
        
    def configure(self, arg_list=[]):
        parser = argparse.ArgumentParser(
            description='Session Manager.')
        
        parser.add_argument('-p', '--parent', nargs='?', default=None, type=int,
                           help="Which parent Session to branch from.")
        parser.add_argument('-r', '--rule', nargs='?', default='reward-max', type=str,
                           help="Which Session to branch from upon finishing.")
        parser.add_argument('-e', '--env-name', nargs='?', default='Test', type=str,
                           help="Gym Environment name to load.")
        parser.add_argument('-ep', '--episodes', nargs='?', default=100, type=int,
                           help="Number of episodes to run.")
        
        parser.parse_args(arg_list, namespace=self)
        self.configured = True
        
    def _get_session_from_rule(self):
        print(f"_get_parent_from_rule: {self.rule}")
        if self.rule == 'reward-max':
            query = self.db.query(Session).filter(
                Session.reward_mean != None).order_by(
                desc(Session.reward_mean)).all()
            
            for session in query:
                if len(session.children) < 3:
                    return session
            return None
        else:
            raise NameError(f"Rule {self.rule} is not defined.")
        
    def initialize_session(self):
        self.connect_to_database()
        
        if self.parent is not None:
            parent = self.db.query(Session).get(self.parent)
        else:
            parent = self._get_session_from_rule()
            
        if parent is None:
            iteration = 0
        else:
            iteration = parent.iteration + 1
            
        # create or fetch the Agent
        if parent is None:
            env = environment_parser(self.env_name)
            config = {}
            agent_instance = ActorCriticAgent(env)
            agent = Agent(
                config=json.dumps(config, cls=NumpyEncoder),
                weights=json.dumps(agent_instance.model.get_weights(), cls=NumpyEncoder)
            )
            self.db.add(agent)
        else:
            parent_agent = parent.agent(full=True)
            agent = Agent(
                config=json.dumps(parent_agent['config'], cls=NumpyEncoder),
                weights=json.dumps(parent_agent['weights'], cls=NumpyEncoder)
            )
            self.db.add(agent)
            
        try:
            repo = Repo('./')
            git_url = [url for url in repo.remote().urls][0]
            commit = repo.head.object.hexsha
        except:
            git_url = None
            commit = None
            
        session = Session(parent=parent, agent=agent, env_name=self.env_name, iteration=iteration,
                               episode_iterations=self.episodes, commit=commit, git_url=git_url)
        self.db.add(session)
        self.session_id = session.id
        self.db.commit()
        self.disconnect_from_database()
        
    def train(self):
        print("Training...")
        self.connect_to_database()
        session = self.db.query(Session).get(self.session_id)
        # get the last episode
        last_episode = self.db.query(Episode).filter(
            Episode.session == session).order_by(Episode.id.desc()).first()
        if last_episode is None:
            start_episode = 0
        else:
            start_episode = last_episode.iteration + 1
            
        # setup the environment
        env = environment_parser(self.env_name)
        
        # reconstitute the agent
        agent_config = json.loads(session.agent.config)
        agent_weights = json.loads(session.agent.weights)
        agent = ActorCriticAgent(env=env, **agent_config)
        agent.model.set_weights(agent_weights)
        
        batch = {
            "observations": [],
            "actions": [],
            "discounted_rewards": []
        }
        
        episode_iteration = start_episode
        while episode_iteration < start_episode + session.episode_iterations and len(batch['observations']) > 0:
            episode = Episode(session=session, iteration=episode_iteration, total_reward=0)
            
            observations, actions, rewards, dones, infos = self.run(agent, env, episode)
            
            episode.observations = json.dumps(observations, cls=NumpyEncoder)
            episode.actions = json.dumps(actions, cls=NumpyEncoder)
            episode.rewards = json.dumps(rewards, cls=NumpyEncoder)
            episode.dones = json.dumps(dones, cls=NumpyEncoder)
            episode.infos = json.dumps(infos, cls=NumpyEncoder)
            episode.total_reward = float(np.sum(rewards))
            
            total_rewards = [ep.total_reward for ep in session.episodes]
            session.reward_mean = float(np.mean(total_rewards))
            session.reward_median = float(np.median(total_rewards))
            session.reward_max = float(np.max(total_rewards))
            session.reward_min = float(np.min(total_rewards))
            session.reward_std = float(np.std(total_rewards))
            
            episode.updated_at = datetime.datetime.utcnow()
            session.updated_at = datetime.datetime.utcnow()

            batch['observations'] += observations
            batch['actions'] += actions
            batch['discounted_rewards'] += get_discounted_rewards(rewards, agent.gamma)
            episode_iteration += 1

            if len(batch['observations']) >= agent.batch_size:
                for epoch in range(agent.epochs):
                    agent.learn(**batch)
                batch = {
                    "observations": [],
                    "actions": [],
                    "discounted_rewards": []
                }
                session.agent.weights = json.dumps(agent.model.get_weights(), cls=NumpyEncoder)
                trained_once = True
                
            self.db.commit()
                
        self.db.commit()
        self.disconnect_from_database()
                
    def run(self, agent, env, episode):
        obs = env.reset()
        done = False
        rewards = []
        actions = []
        observations = []
        dones = []
        infos = []

        step_iteration = 0
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
                
            if done:
                return observations, actions, rewards, dones, infos
            
            obs = next_obs
            step_iteration += 1

class Agent(Base):
    __tablename__ = 'agents'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    summary = Column(String)
    save = Column(Boolean, default=False)
    config = Column(JSON)
    weights = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"Agent {self.id}"

    def __str__(self):
        return f"Agent {self.id}"
        
    def __call__(self, *args, **kargs):
        data = {
            "id": self.id,
            "name": self.name,
            "summary": self.summary,
            "config": json.loads(self.config),
            "save": self.save,
            "created_at": str(self.created_at)
        }
        
        try:
            if kargs['full']:
                data["weights"] = json.loads(self.weights)
        except:
            pass
        
        return data
    
class Session(Base):
    __tablename__ = 'sessions'

    id = Column(Integer, primary_key=True)
    children = relationship("Session", backref=backref('parent', remote_side=[id]))
    parent_id = Column(Integer, ForeignKey('sessions.id'))
    
    episodes = relationship("Episode", backref="session")
    
    agent_id = Column(Integer, ForeignKey('agents.id'))
    agent = relationship("Agent", backref="agent")
    
    iteration = Column(Integer)
    save = Column(Boolean, default=False)
    env_name = Column(String)
    commit = Column(String)
    git_url = Column(String)
    episode_iterations = Column(Integer)
    
    reward_mean = Column(Float)
    reward_median = Column(Float)
    reward_std = Column(Float)
    reward_max = Column(Float)
    reward_min = Column(Float)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime)

    def __repr__(self):
        return f"Session {self.id}: {self.iteration}"

    def __str__(self):
        return f"Session {self.id}: {self.iteration}"
        
    def __call__(self, *args, **kargs):
        data = {
            "id": self.id,
            "agent_id": self.agent_id,
            "iteration": self.iteration,
            "parent_id": self.parent_id,
            "env_name": self.env_name,
            "commit": self.commit,
            "git_url": self.git_url,
            "reward_mean": self.reward_mean,
            "reward_median": self.reward_median,
            "reward_std": self.reward_std,
            "reward_max": self.reward_max,
            "reward_min": self.reward_min,
            "episode_iterations": self.episode_iterations,
            "save": self.save,
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at)
        }
        return data

class Episode(Base):
    __tablename__ = 'episodes'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'))
    iteration = Column(Integer)
    save = Column(Boolean, default=False)
    total_reward = Column(Float)
    
    observations = Column(JSON)
    actions = Column(JSON)
    rewards = Column(JSON)
    dones = Column(JSON)
    infos = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime)

    def __repr__(self):
        return f"Episode {self.id}: {self.iteration}"

    def __str__(self):
        return f"Episode {self.id}"
    
    def __str__(self):
        return f"Episode {self.id}"
        
    def __call__(self, *args, **kargs):
        return {
            "id": self.id,
            "iteration": self.iteration,
            "session_id": self.session_id,
            "total_reward": self.total_reward,
            "save": self.save,
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at)
        }
        
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

def environment_parser(env_name):
    if env_name == 'Test':
        return TestingEnvironment()
    else:
        return gym.make(env_name)

class TestingEnvironment:
    def __init__(self, steps=10):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(1,))
        self.steps = steps
    
    def reset(self):
        self.mean = np.random.choice(3) - 1.
        self.current_step = 0
        return np.array([np.random.normal(loc=self.mean)])
    
    def step(self, action):
        action -= 1.
        if action == self.mean:
            reward = 1.
        elif np.abs(action - self.mean) == 1.:
            reward = 0.
        else:
            reward = -1.
        
        self.current_step += 1
        if self.current_step >= self.steps:
            done = True
        else:
            done = False
            
        return np.array([np.random.normal(loc=self.mean)]), reward, done, {}
        