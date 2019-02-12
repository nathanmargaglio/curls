import sqlalchemy as db
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, JSON, DateTime
import datetime
import os
import json
import time
import numpy as np

class Logger:
    def __init__(self, path="mdp/", memory=False):
        if memory:
            self.engine = db.create_engine('sqlite://')
        else:
            os.makedirs(path, exist_ok=True)
            self.engine = db.create_engine(f'sqlite:///{path}{int(time.time())}.db')

        self.connection = self.engine.connect()
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        self.tables = {}
        self.tag_meta = {}
        self.last_called = None

    def get_orm(self, tablename):
        Base = declarative_base()
        class Table(Base):
            __tablename__ = tablename

            id = Column(Integer, primary_key=True)
            tag = Column(String)
            group = Column(String)
            data = Column(String)
            created_at = Column(DateTime, default=datetime.datetime.utcnow)
            
            def __repr__(self):
                grp = self.group if self.group is not None else ''
                eli = '...' if len(self.data) > 64 else ''
                return f"{self.created_at} - {grp}/{self.tag} : {self.data[:64]}{eli}"
            
            def __str__(self):
                s = f"id    : {self.id}\n"
                s += f"tag   : {self.tag}\n"
                s += f"group : {self.group}\n"
                s += f"data  : {self.data}\n"
                s += "c_at  : (self.created_at)\n"
                return s
            
            def get_data(self):
                return json.loads(self.data)
            
            def __call__(self):
                return self.get_data()

        Base.metadata.create_all(self.engine)
        return Table
    
    def generate_table(self, tablename):
        table = self.get_orm(tablename)
        self.tables[tablename] = table
        return table
    
    def get_table(self, tablename):
        if tablename not in self.tables:
            return self.generate_table(tablename)
        else:
            return self.tables[tablename]
        
    def register_tag(self, tag, tablename, group=None):
        if tag in self.tag_meta:
            return False
        
        self.tag_meta[tag] = {
            "table": self.get_table(tablename),
            "group": group
        }
        return True
    
    def encode_data(self, obj):
        if type(obj.data) != str:
            obj.data = json.dumps(data, cls=NumpyEncoder)
        return obj
    
    def decode_data(self, obj):
        if type(obj.data) == str:
            obj.data = json.loads(obj.data)
        return obj
    
    def tail(self, tag, limit=10):
        assert tag in self.tag_meta, "Register tag first: .register_tag(tag, tablename)"
        table = self.tag_meta[tag]['table']
        return self.session.query(table).order_by(table.id.desc()).limit(limit).all()[::-1]
    
    def last(self, tag, limit=10, update_last=True):
        assert tag in self.tag_meta, "Register tag first: .register_tag(tag, tablename)"
        table = self.tag_meta[tag]['table']
        self.last_called = self.session.query(table).order_by(table.id.desc()).first() if update_last else self.last_called
        return self.last_called
    
    def since_last(self, tag, limit=10, update_last=True):
        assert tag in self.tag_meta, "Register tag first: .register_tag(tag, tablename)"
        
        table = self.tag_meta[tag]['table']
        since_last_query = self.session.query(table).order_by(table.id.desc())
        if self.last_called is not None:
            since_last_query = since_last_query.filter(table.id > self.last_called.id)
            
        since_last = since_last_query.limit(limit).all()[::-1]
        self.last_called = since_last[-1] if update_last and len(since_last) else self.last_called
        return since_last
    
    def rollback(self):
        return self.session.rollback()
    
    def __call__(self, tag, data):
        if tag not in self.tag_meta:
            self.register_tag(tag, 'meta')
            
        table = self.tag_meta[tag]['table']
        group = self.tag_meta[tag]['group']
        
        t = table(tag=tag, group=group, data=json.dumps(data, cls=NumpyEncoder))
        self.session.add(t)
        self.session.commit()
        return t
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)