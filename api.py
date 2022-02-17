import json
import flask
import flask_cors
import flask_restful

app = flask.Flask(__name__)
flask_cors.CORS(app)
api = flask_restful.Api(app)


class DummyList(flask_restful.Resource):

    @staticmethod
    def get():
        parameter = flask.request.args.get('parameter')
        return json.loads({'coucou': parameter, 'hello': 'yellow'})


api.add_resource(DummyList, "/list")

if __name__ == '__main__':
    app.run(debug=True)
