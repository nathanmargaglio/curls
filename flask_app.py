from flask import Flask
from flask_restful import Resource, Api
from MarkovDecisionProcess import MarkovDecisionProcess

app = Flask(__name__)
api = Api(app)

class GetData(Resource):
    def get(self):
        mdp = MarkovDecisionProcess()
        tail = mdp.get_episode()
        if tail is not None:
            data = [t() for t in tail]
            rewards = [0]
            steps = []
            for d in data:
                print(d)
                rewards.append(d['reward'] + rewards[-1])
                steps.append(d['step'])
            steps.append(steps[-1] + 1)
            return {
                'data': data,
                'meta': {
                    'rewards': rewards,
                    'steps': steps
                }
            }
        else:
            return {'data': {}}

api.add_resource(GetData, '/api')

if __name__ == '__main__':
    app.run(debug=True)