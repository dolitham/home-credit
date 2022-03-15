import pickle
import re
import time
from json import loads, dumps
import flask
import flask_cors
import flask_restful
import numpy as np
from lime import lime_tabular

app = flask.Flask(__name__)
flask_cors.CORS(app)
api = flask_restful.Api(app)
target_column = 'Loan Status'

df_for_model = pickle.load(open('df_for_api', 'rb')).rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
df_for_user = pickle.load(open('df_pretty_for_api', 'rb'))
model = pickle.load(open('model_for_api', 'rb'))


def numpy_int_format_remover(value):
    return int(value) if type(value) in [np.uint8, np.int64] else value


def convert_dict(dico):
    return {key: numpy_int_format_remover(value) for key, value in dico.items()}


@app.route('/')
def index():
    return "<h1>Hello</h1>"


class PredictClient(flask_restful.Resource):

    @staticmethod
    def get():
        client_id = flask.request.args.get('client_id')
        X = df_for_model.drop(columns=['TARGET', target_column], errors='ignore')
        readable_names = df_for_user.drop(columns=['TARGET', target_column], errors='ignore').columns
        to_predict = X.loc[int(client_id)]
        t0 = time.time()
        prediction = model.predict_proba(np.array(to_predict).reshape(1, -1))
        explainer = lime_tabular.LimeTabularExplainer(X.values, feature_names=readable_names)
        explanation = explainer.explain_instance(to_predict, model.predict_proba, num_features=len(readable_names) // 2)
        return {'STATUS': 'success', 'prediction': float(prediction[0][1]),
                'runtime': "{:.2f}s".format(time.time() - t0),
                'explanation': explanation.as_html(predict_proba=False)}


class GetClientData(flask_restful.Resource):

    @staticmethod
    def get():
        client_id = flask.request.args.get('client_id')
        try:
            client_id = int(client_id)
        except ValueError:
            return loads(dumps({'STATUS': 'fail', 'message': 'invalid client ID'}))
        try:
            client_data_series = df_for_user.loc[int(client_id)]
        except KeyError:
            return loads(dumps({'STATUS': 'fail', 'message': 'client not found'}))
        return dumps({'STATUS': 'success', 'data': convert_dict(client_data_series.to_dict())})


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
        filtered_df = df_for_user
        if active_filters:
            for feature in active_filters:
                filtered_df = filter_df_by_feature(filtered_df, feature, active_filters[feature])
        return dumps(
            {'client_ids': list(filtered_df.index), 'target': filtered_df[target_column].value_counts().to_dict()})


class GetSetFilter(flask_restful.Resource):

    @staticmethod
    def get():
        feature = flask.request.args.get('feature')
        if feature not in df_for_user.columns:
            return {'STATUS': 'fail', 'message': 'Invalid feature'}
        values = df_for_user[feature]
        return dumps({'STATUS': 'success', 'dtype': str(values.dtype), 'values': list(set(values))})


class GetFeaturesList(flask_restful.Resource):

    @staticmethod
    def get():
        return dumps(list(df_for_user.columns))


class GetFeatureData(flask_restful.Resource):

    @staticmethod
    def get():
        feature_requested = flask.request.args.get('feature')
        active_filters = flask.request.json
        filtered_df = df_for_user
        if active_filters:
            for feature in active_filters:
                filtered_df = filter_df_by_feature(filtered_df, feature, active_filters[feature])
        filtered_df = filtered_df[filtered_df[target_column] != 'Unknown']
        feature_data = {'feature': list(filtered_df[feature_requested]),
                        'feature_type': str(filtered_df[feature_requested].dtype),
                        'TARGET': list(filtered_df[target_column])}
        return dumps(feature_data)


api.add_resource(GetClientData, "/client_data")
api.add_resource(PredictClient, "/predict")
api.add_resource(GetClientIdList, "/client_id_list")
api.add_resource(GetFeaturesList, "/features_list")
api.add_resource(GetSetFilter, "/feature")
api.add_resource(GetFeatureData, "/feature_data")

if __name__ == '__main__':
    app.run(debug=True)
