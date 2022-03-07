import streamlit as st
from app_requests import *
import matplotlib.pyplot as plt
import seaborn as sns

#  streamlit run /Users/julie/PycharmProjects/home-credit/app.py

# %% Init

st.set_page_config(page_title='Prêt à dépenser', page_icon=":bank:", layout='wide', initial_sidebar_state = 'auto')

all_features = get_features_list()
value = get_dummy_val()

textbox = st.empty()  # TODO delete this
textbox.info(value)

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
    st.write(nb_available_clients, 'client' + ('s'*(nb_available_clients>1)) + ' available')
    index_client = st.select_slider(options=[None] + list_client_ids, label='Client ID')
    valid_client_data = False

    if index_client is not None:
        valid_prediction, value_prediction = get_prediction_client(int(index_client))
        if valid_prediction:
            st.write('prediction', value_prediction)
        else:
            st.write('could not predict')

        valid_client_data, client_data = get_client_data(index_client)

    buttons = dict()
    columns = st.columns([1, 1, 1, 1, 1])
    for i, feature in enumerate(all_features[:5]):
        buttons[feature] = columns[i].checkbox(feature, disabled=(feature in active_filters.keys()))

    f, (ax1, ax2) = plt.subplots(1, 2)
    for feature in {feature for feature in buttons if buttons[feature]}:
        feature_data = get_feature_data(feature, active_filters)

        st.write(feature, feature_data['feature_type'])

        if feature_data['feature_type'] == 'float64':
            sns.kdeplot(x=feature_data['feature'], hue=feature_data['TARGET'], ax=ax1)
            ax1.set_yticklabels([])
            if valid_client_data:
                # noinspection PyUnboundLocalVariable
                ax1.axvline(x=client_data[feature], color="red", ls="--", lw=2.5)
        else:
            sns.histplot(x=feature_data['feature'], hue=feature_data['TARGET'], ax=ax2)
            ax2.set_yticklabels([])
            if valid_client_data:
                ax2.axvline(x=client_data[feature], color="red", ls="--", lw=2.5)

    st.pyplot(f)

else:
    st.write('No client found')
