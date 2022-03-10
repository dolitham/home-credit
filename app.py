import pickle
import re
import time
from json import loads, dumps
import flask
import flask_cors
import flask_restful
import numpy as np

app = flask.Flask(__name__)
flask_cors.CORS(app)
api = flask_restful.Api(app)

# id_column = 'SK_ID_CURR'
df_for_model = pickle.load(open('df', 'rb'))
df_for_model = df_for_model.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
model = pickle.load(open('lgbm_model', 'rb'))


@app.route('/')
def index():
    return "<h1>Hello</h1>"


class PredictClient(flask_restful.Resource):

    @staticmethod
    def get():
        client_id = flask.request.args.get('client_id')
        to_predict = df_for_model[model.feature_name_].loc[int(client_id)]
        t0 = time.time()
        prediction = model.predict_proba(np.array(to_predict).reshape(1, -1))
        return {'STATUS': 'success', 'prediction': float(prediction[0][1]),
                'runtime': "{:.2f}s".format(time.time() - t0)}


class GetClientData(flask_restful.Resource):

    @staticmethod
    def get():
        client_id = flask.request.args.get('client_id')
        try:
            client_id = int(client_id)
        except ValueError:
            return loads(dumps({'STATUS': 'fail', 'message': 'invalid client ID'}))

        client_data_df = df_for_model[model.feature_name_].loc[int(client_id)]

        if client_data_df.shape[0] < 1:
            return loads(dumps({'STATUS': 'fail', 'message': 'client not found'}))

        return {'STATUS': 'success', 'data': dumps(client_data_df.to_dict())}


def filter_df_by_feature(df, feature, values):
    if values[0] == 'float64':
        feature_type, (mini, maxi) = values
        return df[(mini <= df[feature]) & (df[feature] <= maxi)]

    feature_type, selected_values = values
    return df[df[feature].isin(selected_values)]


class GetClientIdList(flask_restful.Resource):

    @staticmethod
    def get():
        active_filters = flask.request.json
        filtered_df = df_for_model
        if active_filters:
            for feature in active_filters:
                filtered_df = filter_df_by_feature(filtered_df, feature, active_filters[feature])
        return dumps(list(filtered_df.index))


class GetSetFilter(flask_restful.Resource):

    @staticmethod
    def get():
        feature = flask.request.args.get('feature')
        if feature not in df_for_model.columns:
            return {'STATUS': 'fail', 'message': 'Invalid feature'}
        values = df_for_model[feature]
        return dumps({'STATUS': 'success', 'dtype': str(values.dtype), 'values': list(set(values))})


class GetFeaturesList(flask_restful.Resource):

    @staticmethod
    def get():
        return dumps(list(df_for_model.columns))


class GetFeatureData(flask_restful.Resource):

    @staticmethod
    def get():
        feature_requested = flask.request.args.get('feature')
        active_filters = flask.request.json
        filtered_df = df_for_model
        for feature in active_filters:
            filtered_df = filter_df_by_feature(filtered_df, feature, active_filters[feature])
        feature_data = {'feature': list(filtered_df[feature_requested]),
                        'feature_type': str(filtered_df[feature_requested].dtype),
                        'TARGET': list(filtered_df['TARGET'])}
        return dumps(feature_data)


api.add_resource(GetClientData, "/client_data")
api.add_resource(PredictClient, "/predict")
api.add_resource(GetClientIdList, "/client_id_list")
api.add_resource(GetFeaturesList, "/features_list")
api.add_resource(GetSetFilter, "/feature")
api.add_resource(GetFeatureData, "/feature_data")


if __name__ == '__main__':
    app.run(debug=True)

