from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from pprint import pprint
import numpy as np
from SessionManager import SessionManager, Session, Episode, Step, Agent
from sqlalchemy import desc, asc

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('from')
parser.add_argument('to')
parser.add_argument('episode')

entity_map = {
    "sessions": Session,
    "episodes": Episode,
    "steps": Step,
    "agents": Agent
}

class EntityController(Resource):
    sm = SessionManager()
    def get(self, entity, entity_id=None):
        args = parser.parse_args()
        _from = args['from']
        _to = args['to']
        _episode = args['episode']
        
        entity = entity_map[entity]
        if entity_id is None:
            entities_query = self.sm.db.query(entity)
            if _to is not None:
                entities_query = entities_query.filter(entity.iteration <= _to)
            if _from is not None:
                entities_query = entities_query.filter(entity.iteration >= _from)
            if _episode is not None:
                entities_query = entities_query.filter(entity.episode_id == _episode)
            entities = entities_query.limit(100).all()
            return { "data" : [e() for e in entities] }
        else:
            return { "data" : self.sm.db.query(entity).get(entity_id)() }

    def post(self):
        args = parser.parse_args()
        todo_id = int(max(TODOS.keys()).lstrip('todo')) + 1
        todo_id = 'todo%i' % todo_id
        TODOS[todo_id] = {'task': args['task']}
        return TODOS[todo_id], 201
    
    def put(self, entity_id):
        args = parser.parse_args()
        task = {'task': args['task']}
        TODOS[entity_id] = task
        return task, 201

api.add_resource(EntityController, '/api/<entity>', endpoint="entity")
api.add_resource(EntityController, '/api/<entity>/<entity_id>', endpoint="entities")

if __name__ == '__main__':
    app.run(debug=True)