import pickle
import seaborn as sns
import matplotlib.pyplot as plt

df_for_model = pickle.load(open('df', 'rb'))
model = pickle.load(open('lgbm_model', 'rb'))
probas = model.predict_proba(df_for_model.drop(columns=['TARGET', 'percentage_columns_null']))

targets = df_for_model['TARGET']
predictions = probas[:, 1]
predict_class = model.predict(df_for_model.drop(columns=['TARGET', 'percentage_columns_null']))

sns.boxplot(x=targets, y=predictions)
plt.show()
