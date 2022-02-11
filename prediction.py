from kaggle_kernel import preprocess_data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kaggle_kernel import timer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# %%

filename = 'df.csv'

try:
    with timer('fetching data'):
        df = pd.read_csv(filename, index_col=0).drop(columns=['index'], errors='ignore')
        print('df shape', df.shape)
except FileNotFoundError:
    df = preprocess_data()
    with(timer('saving data')):
        df.to_csv(filename, index=True)

# %%

str_len = max([len(c) for c in df.columns])
for c in df.columns:
    null_percentage_this_col = 100 * df[c].isnull().sum() // len(df[c])
    print(c.rjust(str_len), null_percentage_this_col, '% null')

df['percentage_columns_null'] = (100 * df.isnull().sum(axis=1) / df.shape[1]).astype(int)
df['TARGET'] = df['TARGET'].replace(np.NaN, 'nan')

df_w_target = df[df['TARGET'] != 'nan']

sns.histplot(data=df, x="percentage_columns_null", hue="TARGET", multiple="stack", bins=38)
plt.title('Number of individuals per null_columns')
plt.show()

sns.histplot(data=df, x="percentage_columns_null", hue="TARGET", multiple="stack", bins=38, log_scale=(False, True))
plt.title('Number of individuals per null_columns (log scale)')
plt.show()

# %%

sns.histplot((100 * df.isnull().sum(axis=0) / df.shape[0]))
plt.xlabel('percentage null values')
plt.ylabel('columns count')
plt.title('columns filling')
plt.show()

# %%
X = df_w_target.loc[:, df_w_target.isnull().sum(axis=0) == 0].drop(columns=['TARGET', 'percentage_columns_null'])
y = df_w_target.loc[:, 'TARGET'].astype(int)

print(X.shape, y.shape)

# %%

sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X, y)

# %%
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

# %%

models = dict()


def train_and_predict(smote_state, model):
    model_name = model.__str__().split('(')[0]
    print(model_name, smote_state)
    my_X, my_y = (X, y) if smote_state == 'without SMOTE' else (X_sm, y_sm)
    X_train, X_test, y_train, y_test = train_test_split(my_X, my_y, test_size=0.2, random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print(f'Accuracy = {accuracy:.2f}\nRecall = {recall:.2f}\n')
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    plt.title('Confusion Matrix ' + model_name + ' ' + smote_state)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.show()

    models[model_name + ' ' + smote_state] = model


# %%

lr = LogisticRegression(random_state=0)
train_and_predict('without SMOTE', lr)

# %%

lr = LogisticRegression(random_state=0)
train_and_predict('with SMOTE', lr)

# %%

dt = DecisionTreeClassifier(random_state=0)
train_and_predict('without SMOTE', dt)

# %%

dt = DecisionTreeClassifier(random_state=0)
train_and_predict('with SMOTE', dt)

# %%
