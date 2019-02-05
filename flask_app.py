from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from pprint import pprint
import numpy as np
from SessionManager import SessionManager, Session, Episode, Step, Agent
from sqlalchemy import desc, asc, func

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('from', type=int)
parser.add_argument('to', type=int)
parser.add_argument('page', type=int)
parser.add_argument('per_page', type=int)

entity_map = {
    "sessions": Session,
    "episodes": Episode,
    "steps": Step,
    "agents": Agent
}

def parse_pagination(args):
    if type(args['page']) == int:
        page = args['page']
        if page < 1:
            page = 1
    else:
        page = 1
        
    if type(args['per_page']) == int:
        per_page = int(args['per_page'])
        if per_page < 1:
            per_page = 1
        if per_page > 100:
            per_page = 100
    else:
        per_page = 100
        
    return page, per_page

def get_count(q):
    count_q = q.statement.with_only_columns([func.count()]).order_by(None)
    count = q.session.execute(count_q).scalar()
    return count

class EntityController(Resource):
    sm = SessionManager()
    def get(self, entity, entity_id=None, subentity=None):
        args = parser.parse_args()
        _from = args['from']
        _to = args['to']
        page, per_page = parse_pagination(args)
        
        if entity_id is None:
            entity = entity_map[entity]
            entities_query = self.sm.db.query(entity)
            if _to is not None:
                entities_query = entities_query.filter(entity.iteration <= _to)
            if _from is not None:
                entities_query = entities_query.filter(entity.iteration >= _from)
                
            entities = entities_query.order_by(entity.id).offset(per_page * (page - 1)).limit(per_page).all()
            return { "data" : [e() for e in entities], "page":  page, "per_page": per_page, "count": get_count(entities_query)}
        elif subentity is not None:
            subentities_query = self.generate_subquery(entity, entity_id, subentity)

            entity = entity_map[entity]
            subentity = entity_map[subentity]
            
            if _to is not None:
                subentities_query = subentities_query.filter(subentity.iteration <= _to)
            if _from is not None:
                subentities_query = subentities_query.filter(subentity.iteration >= _from)
                
            subentities = subentities_query.order_by(subentity.id).offset(per_page * (page - 1)).limit(per_page).all()
            return { "data" : [e() for e in subentities], "page":  page, "per_page": per_page, "count": get_count(subentities_query)}
        else:
            entity = entity_map[entity]
            return { "data" : self.sm.db.query(entity).get(entity_id)() }

    def generate_subquery(self, entity, entity_id, subentity):
        if entity == 'sessions':
            if subentity == 'episodes':
                subentity = entity_map[subentity]
                return self.sm.db.query(subentity).filter(subentity.session_id == entity_id)
            if subentity == 'agents':
                subentity = entity_map[subentity]
                entity = entity_map[entity]
                entity = self.sm.db.query(entity).get(entity_id)()
                return self.sm.db.query(subentity).filter(subentity.id == entity['agent_id'])
        if entity == 'episodes':
            if subentity == 'steps':
                subentity = entity_map[subentity]
                return self.sm.db.query(subentity).filter(subentity.episode_id == entity_id)
            
        abort(400, message="Query not yet supported.")
        
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

api.add_resource(EntityController, '/api/<entity>', endpoint="entities")
api.add_resource(EntityController, '/api/<entity>/<entity_id>', endpoint="entity")
api.add_resource(EntityController, '/api/<entity>/<entity_id>/<subentity>', endpoint="subentities")

if __name__ == '__main__':
    app.run(debug=True)