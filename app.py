import math

import streamlit as st
from app_requests import *
import matplotlib.pyplot as plt
import seaborn as sns

#  streamlit run /Users/julie/PycharmProjects/home-credit/app.py

# %% Init

st.set_page_config(page_title='Prêt à dépenser', page_icon=":bank:", layout='wide', initial_sidebar_state = 'auto')

all_features = get_features_list()

textbox = st.empty()  # TODO delete this

# %% Sidebar filtering

selected_features = st.sidebar.multiselect(label='Filter by :', options=all_features)

filters = dict()
active_filters = dict()

for feature in selected_features:
    feature_type, feature_values = get_possible_values(feature)
    if feature_type == 'float64':
        filters[feature] = st.sidebar.slider(label=feature, value=(min(feature_values), max(feature_values)))
        if filters[feature] != (min(feature_values), max(feature_values)):
            active_filters[feature] = (feature_type, filters[feature])
    else:
        filters[feature] = st.sidebar.multiselect(options=feature_values, label=feature)
        if set(filters[feature]) not in [set(feature_values), set()]:
            active_filters[feature] = (feature_type, filters[feature])

textbox.info(active_filters)

# %% Content

list_client_ids = get_list_client_ids(active_filters)

nb_available_clients = len(list_client_ids)
if nb_available_clients >= 1:
    st.write(nb_available_clients, 'client' + ('s'*(nb_available_clients>1)) + ' matching filters found')
    index_client = st.select_slider(options=[None] + list_client_ids, label='Select a client')
    valid_client_data = False

    if index_client is not None:
        valid_prediction, value_prediction = get_prediction_client(int(index_client))
        if valid_prediction:
            st.write('Bankrupcy prediction : ', '{:.2f}%'.format(100*(value_prediction)))
        else:
            st.write('could not predict')

        valid_client_data, client_data = get_client_data(index_client)

    buttons = dict()

    for i, feature in enumerate(all_features[:5]):
        buttons[feature] = st.sidebar.checkbox(feature, disabled=(feature in active_filters.keys()))

    features_chosen = [feature for feature in buttons if buttons[feature]]
    nb_features_chosen = len(features_chosen)

    if len(features_chosen):
        nb_plot_rows = int(math.ceil(nb_features_chosen / 2))
        f, ax = plt.subplots(nb_plot_rows, 2)
        f.set_figheight(5*nb_plot_rows)
        f.set_figwidth(15)

        for feature_index, feature in enumerate(features_chosen):
            index_col = feature_index % 2
            index_row = feature_index // 2
            this_ax = ax[index_col] if nb_plot_rows == 1 else ax[index_row][index_col]
            feature_data = get_feature_data(feature, active_filters)

            if feature_data['feature_type'] == 'float64':
                sns.kdeplot(x=feature_data['feature'], hue=feature_data['TARGET'], ax=this_ax)
            else:
                sns.histplot(x=feature_data['feature'], hue=feature_data['TARGET'], ax=this_ax,
                             multiple="dodge", discrete=True, shrink=.8)
                if set(feature_data['feature']) == {0, 1}:
                    this_ax.set_xticks([0, 1], [0, 1])

            this_ax.set_yticklabels([])
            this_ax.set_title(feature)
            if valid_client_data:
                # noinspection PyUnboundLocalVariable
                this_ax.axvline(x=client_data[feature], color="red", ls="--", lw=2.5)

        st.pyplot(f)

else:
    st.write('No client found')
