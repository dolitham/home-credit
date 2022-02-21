from json import loads, dumps
import flask
import flask_cors
import flask_restful
from pandas import read_csv

app = flask.Flask(__name__)
flask_cors.CORS(app)
api = flask_restful.Api(app)

df = read_csv('application_train_extract.csv')
id_column = 'SK_ID_CURR'


class GetClientData(flask_restful.Resource):

    @staticmethod
    def get():
        client_id = flask.request.args.get('client_id')
        try:
            client_id = int(client_id)
        except ValueError:
            return loads(dumps({'STATUS': 'fail', 'message': 'invalid client ID'}))

        client_data_df = df[df[id_column] == client_id]
        if client_data_df.shape[0] < 1:
            return loads(dumps({'STATUS': 'fail', 'message': 'client not found'}))

        client_data_df = df[df[id_column] == client_id]
        if client_data_df.shape[0] < 1:
            return loads(dumps({'STATUS': 'fail', 'message': 'client not found'}))

        return loads(dumps({'STATUS': 'success', 'data': dumps(client_data_df.iloc[0].to_dict())}))


api.add_resource(GetClientData, "/client_data")

if __name__ == '__main__':
    app.run(debug=True)
