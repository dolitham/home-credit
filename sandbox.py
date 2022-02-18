import pandas as pd

train = pd.read_csv('input/application_train.csv')

columns = ['AMT_INCOME_TOTAL', 'TARGET', 'SK_ID_CURR']

app_train = train[columns]
print(app_train.shape)
app_train.to_csv('application_train_extract.csv')