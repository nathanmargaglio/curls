from flask import Flask
from flask_restful import Resource, Api
from MarkovDecisionProcess import MarkovDecisionProcess

app = Flask(__name__)
api = Api(app)

class GetData(Resource):
    def get(self):
        mdp = MarkovDecisionProcess()
        last = mdp.last()
        if last is not None:
            return {'data': last()}
        else:
            return {'data': {}}

api.add_resource(GetData, '/')

if __name__ == '__main__':
    app.run(debug=True)