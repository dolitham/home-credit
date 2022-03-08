import json
import requests


url = 'http://127.0.0.1:5000/'
client_id_list_path = 'client_id_list'
prediction_path = 'predict'
client_data_path = 'client_data'
feature_path = 'feature'
feature_data_path = 'feature_data'
features_list_path = 'features_list'
dummy_path = 'dummy'


def get_list_client_ids(active_filters):
    return json.loads(requests.get(url=url + client_id_list_path,
                                            headers={"Content-Type": "application/json"},
                                            data=json.dumps(active_filters)).json())


def get_prediction_client(client_id):
    response = json.loads(requests.get(url=url + prediction_path + '?client_id=' + str(int(client_id))).content)
    return response['STATUS'] == 'success', response['prediction']


def get_client_data(client_id):
    response = json.loads(requests.get(url=url + client_data_path + '?client_id=' + str(int(client_id))).content)
    return response['STATUS'] == 'success', json.loads(response['data'])


def get_possible_values(feature):
    response = json.loads(requests.get(url=url + feature_path + '?feature=' + feature).json())
    if response['STATUS'] == 'success':
        return response['dtype'], set(response['values'])
    return None, []


def get_feature_data(feature, active_filters):
    return json.loads(requests.get(url=url + feature_data_path + '?feature=' + feature,
                                   headers={"Content-Type": "application/json"},
                                   data=json.dumps(active_filters)).json())


def get_features_list():
    return json.loads(requests.get(url=url + features_list_path).json())
