import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import plotly.graph_objects as go
import pandas as pd

pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 999)

df = pd.read_csv('WHS8_110.csv')

new_columns = {
    'Country': 'country'
}

df.columns = df.iloc[0].values
df = df[1:]
df = df.fillna(0).copy()
df.columns = df.columns.astype(str)
df.columns = [col.replace('.0', '') for col in df.columns]
df.rename(columns=new_columns, inplace=True)

# df_country = df.head(30)
# df_year = df.head(30)

app = dash.Dash()

server = app.server

app.layout = html.Div(
    [
        html.H1(
            'Dummy Dashboard',
            style={'text-align': 'center'}
        ),
        html.Div([
            dcc.Dropdown(
                id='year_selection',
                options=[
                    {'label': '2019', 'value': '2019'},
                    {'label': '2018', 'value': '2018'},
                    {'label': '2017', 'value': '2017'},
                    {'label': '2016', 'value': '2016'},
                    {'label': '2015', 'value': '2015'},
                    {'label': '2014', 'value': '2014'},
                    {'label': '2013', 'value': '2013'},
                    {'label': '2012', 'value': '2012'}
                ],
                value='2019'
            ),
            dcc.Graph(id='bargraph'),
            dcc.Dropdown(
                id='country_selection',
                options=[
                    {'label': 'Romania', 'value': 'Romania'},
                    {'label': 'United States of America', 'value': 'United States of America'},
                    {'label': 'France', 'value': 'France'},
                    {'label': 'Hungary', 'value': 'Hungary'},
                    {'label': 'Italy', 'value': 'Italy'},
                    {'label': 'Sudan', 'value': 'Sudan'},
                    {'label': 'Spain', 'value': 'Spain'},
                    {'label': 'Egypt', 'value': 'Egypt'},
                    {'label': 'India', 'value': 'India'},
                ],
                value='Romania'
            ),
            dcc.Graph(id='linegraph'),
        ],
            style={'padding': 10})
    ]
)

application_train = pd.read_csv('input/' + 'application_train.csv')

width_income_group = 20000
width_income_group_k = int(width_income_group // 1000)
max_income_group = 20
application_train['income_group'] = application_train['AMT_INCOME_TOTAL'].astype(int) // width_income_group
application_train['income_group'] = application_train['income_group'].apply(lambda x: min(x, max_income_group))

fails = pd.DataFrame(application_train[application_train['TARGET'] == 1]['income_group'].value_counts().sort_index().rename('nb_loans'))
fails['status'] = 'fail'
success = pd.DataFrame(application_train[application_train['TARGET'] == 0]['income_group'].value_counts().sort_index().rename('nb_loans'))
success['status'] = 'success'
to_plot = pd.concat([fails, success])



labels = [str(width_income_group_k * x) + 'k-' + str(width_income_group_k * (x+1)) + 'k' for x in to_plot.index]
max_income = labels[-1]
labels = [((labels[-1].split('-')[0] + '+') if l == max_income else l)for l in labels]
to_plot.index = labels
to_plot = to_plot.reset_index().rename(columns={'index': 'income_group'})


@app.callback(
    Output('bargraph', 'figure'),
    [Input('country_selection', 'value')]
)

def test_chart(country):
    fail = to_plot[to_plot['status'] == 'fail']
    success = to_plot[to_plot['status'] == 'success']
    datapoints = {'data': [go.Bar(x=fail['income_group'], y=fail['nb_loans']), go.Bar(x=success['income_group'], y=success['nb_loans'])],
                  'layout': dict(legend_title_text="Status")}
    return datapoints
#%%

if __name__ == '__main__':
    app.run_server(debug=True)
