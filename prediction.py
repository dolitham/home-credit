import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import warnings
import pickle
import time

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, fbeta_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA

from kaggle_kernel import preprocess_data, timer

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

#%%
temp_directory = 'fast_forward/'


def fast_forward_retrieve(filename):
    return pickle.load(open(temp_directory + filename, 'rb'))


def file_exists(filename):
    return os.path.exists(temp_directory + filename)


def fast_forward_save(obj, filename):
    pickle.dump(obj, open(temp_directory + filename, 'wb'))


#%%

filename = 'df_from_kernel'

if file_exists(filename):
    df = fast_forward_retrieve(filename)
else:
    with timer('fetching data'):
        df = preprocess_data()
    fast_forward_save(df, filename)

#%%

columns = ['TOTALAREA_MODE', 'EMERGENCYSTATE_MODE_No', 'FLAG_OWN_CAR',
           'CODE_GENDER', 'OWN_CAR_AGE', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_2',
           'EXT_SOURCE_3', 'NAME_FAMILY_STATUS_Civil marriage',
           'NAME_FAMILY_STATUS_Married', 'EMERGENCYSTATE_MODE_Yes',
           'NAME_EDUCATION_TYPE_Secondary / secondary special', 'FLAG_EMP_PHONE',
           'NAME_EDUCATION_TYPE_Incomplete higher',
           'NAME_EDUCATION_TYPE_Higher education',
           'WALLSMATERIAL_MODE_Stone, brick', 'PREV_NAME_YIELD_GROUP_high_MEAN',
           'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 'CNT_CHILDREN',
           'WALLSMATERIAL_MODE_Block', 'DAYS_BIRTH'] + ['TARGET']

# df = df.set_index('SK_ID_CURR')[columns]

#%%
df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].astype(int)

#%%

str_len = max([len(c) for c in df.columns])
for c in df.columns:
    null_percentage_this_col = 100 * df[c].isnull().sum() // len(df[c])
    if null_percentage_this_col:
        print(c.rjust(str_len), null_percentage_this_col, '% null')

df['percentage_columns_null'] = (100 * df.isnull().sum(axis=1) / df.shape[1]).astype(int)
df['TARGET'] = df['TARGET'].replace(np.NaN, 'nan')

df_w_target = df[df['TARGET'] != 'nan']
df_w_target.loc[:, 'TARGET'] = df_w_target.loc[:, 'TARGET'].astype(int)

sns.histplot(data=df, x="percentage_columns_null", hue="TARGET", multiple="stack", bins=38)
plt.title('Number of individuals per null_columns')
plt.show()

sns.histplot(data=df, x="percentage_columns_null", hue="TARGET", multiple="stack", bins=38, log_scale=(False, True))
plt.title('Number of individuals per null_columns (log scale)')
plt.show()

#%%

sns.histplot((100 * df.isnull().sum(axis=0) / df.shape[0]))
plt.xlabel('percentage null values')
plt.ylabel('columns count')
plt.title('columns filling')
plt.show()

#%%
X = df_w_target.loc[:, df_w_target.isnull().sum(axis=0) == 0].drop(columns=['TARGET', 'percentage_columns_null'])
y = df_w_target.loc[:, 'TARGET'].astype(int)
print('df w target shape')
print(X.shape, y.shape)

#%%

if file_exists('fast_forward/X_sm') and file_exists('fast_forward/y_sm'):
    X_sm, y_sm = fast_forward_retrieve('fast_forward/X_sm'), fast_forward_retrieve('fast_forward/y_sm')
else:
    with timer('expanding dataset with SMOTE'):
        sm = SMOTE(random_state=0)
        X_sm, y_sm = sm.fit_resample(X, y)
        fast_forward_save(X_sm, 'fast_forward/X_sm')
        fast_forward_save(y_sm, y_sm)
print('shape after smote')
print(X_sm.shape, y_sm.shape)

#%%
"""
n_components = 4
acp = PCA(n_components=n_components)
acp.fit(X)

X_acp = pd.DataFrame(acp.transform(X))
X_sm_acp = pd.DataFrame(acp.transform(X_sm))
X_acp.loc[:, 'TARGET'] = y.astype(int)
X_sm_acp.loc[:, 'TARGET'] = y_sm.astype(int)

for i in range(n_components - 1):
    for j in range(i + 1, n_components):
        f, (ax1, ax2) = plt.subplots(2, 1)
        f.suptitle('Visualisation des individus sur axes de l\'ACP' + str(i) + ' & ' + str(j))

        sns.scatterplot(data=X_acp, x=i, y=j, hue='TARGET', ax=ax1)
        ax1.set_title('avant SMOTE')

        sns.scatterplot(data=X_sm_acp, x=i, y=j, hue='TARGET', ax=ax2)
        ax2.set_title('après SMOTE')
        plt.show()
"""
#%%
# noinspection PyTypeChecker
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Nb customers in each category')

nb_customers = y.value_counts()
sns.barplot(x=nb_customers.index, y=nb_customers.values, ax=ax1)
ax1.set_title('before SMOTE')
ax1.set_ylabel('Number of customers')

nb_customers_sm = y_sm.value_counts()
sns.barplot(x=nb_customers_sm.index, y=nb_customers_sm.values, ax=ax2)
ax2.set_title('after SMOTE')
ax2.set_ylabel('Number of customers')

plt.show()

#%%

models_directory = 'models/'
accuracy_results = dict()
recall_results = dict()
f1_score_results = dict()
f2_score_results = dict()
roc_auc_results = dict()


def save_model(model):
    pickle.dump(model, open(models_directory + model.__str__().split('(')[0], 'wb'))


def retrieve_model(name):
    return pickle.load(open(models_directory + name, 'rb'))


def train_and_predict(use_smote, model):
    id_model = model.__str__().split('(')[0] + ' with SMOTE' if use_smote else ' without SMOTE'
    with timer(id_model):
        my_X, my_y = (X_sm, y_sm) if use_smote else (X, y)
        my_X = my_X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x)).drop(columns=['SK_ID_CURR'],
                                                                                   errors='ignore')
        X_train, X_test, y_train, y_test = train_test_split(my_X, my_y, test_size=0.2, random_state=0)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy_results[id_model] = accuracy_score(y_test, predictions)
        recall_results[id_model] = recall_score(y_test, predictions)
        f1_score_results[id_model] = fbeta_score(y_test, predictions, beta=2)
        f2_score_results[id_model] = fbeta_score(y_test, predictions, beta=1)
        roc_auc_results[id_model] = roc_auc_score(y_test, predictions)

        print(f'Accuracy = {accuracy_results[id_model]:.3f}'
              f'\nRecall = {recall_results[id_model]:.3f}'
              f'\nF1 = {f1_score_results[id_model]:.3f}'
              f'\nF2 = {accuracy_results[id_model]:.3f}'
              f'\nROC AUC = {roc_auc_results[id_model]:.3f}')

        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        plt.title('Confusion Matrix ' + id_model)
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.show()

        save_model(model)

    try:
        print(id_model)
        print(my_X.columns[np.argsort(model.feature_importances_)[::-1][:20]])
    except AttributeError:
        pass

    return predictions


#%%

lr = LogisticRegression(random_state=0)
result_lr = train_and_predict(use_smote=True, model=lr)

#%%

dt = DecisionTreeClassifier(random_state=0)
result_dt = train_and_predict(use_smote=True, model=dt)

#%%
cm_gender = confusion_matrix(df_w_target['TARGET'].astype(int), df_w_target['CODE_GENDER'])
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix GENDER')
sns.heatmap(cm_gender, annot=True, fmt="d", cmap='Blues')
plt.show()

#%%
sns.histplot(data=df_w_target, x="CODE_GENDER", hue="TARGET",
             multiple="fill", bins=2, binwidth=.4)
plt.show()

#%%
cm_own_car = confusion_matrix(df_w_target['TARGET'].astype(int), df_w_target['FLAG_OWN_CAR'])
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix FLAG OWN CAR')
sns.heatmap(cm_own_car, annot=True, fmt="d", cmap='Blues')
plt.show()

#%%
df_w_target.loc[:, 'FLAG_OWN_CAR'] = df_w_target.loc[:, 'FLAG_OWN_CAR'].astype(int).astype(str)

f, (ax1, ax2) = plt.subplots(1, 2)
f.suptitle('Distribution of individuals split by car owning')
sns.histplot(data=df_w_target, x="FLAG_OWN_CAR", hue="TARGET",
             discrete=True, ax=ax1, multiple='stack', shrink=0.9)
ax1.set_title('Number of individuals')
ax2.set_title('Proportion of individuals')
ax1.set_ylabel('')
sns.histplot(data=df_w_target, x="FLAG_OWN_CAR", hue="TARGET",
             discrete=True, multiple='fill', shrink=0.9, ax=ax2)
ax2.set_title('Proportion of individuals')
ax2.set_ylabel('')
plt.show()

#%%

age = (-X_sm['DAYS_BIRTH'] / 365.25).rename('AGE')
sns.displot(x=age, kind="kde", hue=y_sm, fill=True)
plt.show()

#%%

X_sm['CNT_FAM_MEMBERS'] = X_sm['CNT_FAM_MEMBERS'].astype(int)
X_sm['DAYS_BIRTH'] = X_sm['DAYS_BIRTH'].astype(float)

#%%

nb_most_important_columns = 20
index_most_important_columns = np.argsort(dt.feature_importances_)[::-1][:nb_most_important_columns]
most_important_columns = X_sm.columns[index_most_important_columns]
feature_importance = dt.feature_importances_[index_most_important_columns]

for col, importance in zip(most_important_columns, feature_importance):

    f = plt.figure(figsize=(12, 9))
    if X_sm[col].dtype == float:
        sns.displot(x=X_sm[col], kind='kde', hue=y_sm, multiple="layer")
    else:
        sns.displot(x=X_sm[col], kind='hist', shrink=0.8, hue=y_sm, fill=True, multiple="dodge", stat='probability',
                    discrete=True)
    plt.title(f'importance = {importance:.4f}', y=1.0, pad=-14)
    plt.show()

#%%

rf = RandomForestClassifier(random_state=0)
result_rf = train_and_predict(use_smote=True, model=rf)

#%%
sns.displot(x=X['EXT_SOURCE_3'], kind='kde', hue=y, multiple="layer")
plt.show()

#%%

lgbm = LGBMClassifier()
result_lgbm = train_and_predict(use_smote=True, model=lgbm)


#%%

def load_model_and_predict(model_name, prediction_input):
    t0 = time.time()
    model = retrieve_model(model_name)
    prediction = model.predict(prediction_input)
    print("{} - done in {:.2f}s".format('loading model and predicting with ' + model_name, time.time() - t0))
    return prediction


to_predict = X_sm.iloc[0:1, :]
# print(load_model_and_predict('lgbm_model', to_predict))
# print(load_model_and_predict('rf_model', to_predict))

#%%
# pickle.dump(df, open('df', 'wb'))

#%%
