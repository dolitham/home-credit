import pickle
import pandas as pd
import numpy as np
import re

#df_for_model = pickle.load(open('df', 'rb'))
#df_for_model = df_for_model.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
#model = pickle.load(open('lgbm_model', 'rb'))

#client_id = 445221
#model.predict(np.array(df_for_model[model.feature_name_].loc[int(client_id)]).reshape(1, -1))

import shap

df_for_model = pickle.load(open('df', 'rb'))
df_for_model = df_for_model.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
df_for_model_no_target = df_for_model.drop(columns=['TARGET'], errors='ignore')
model = pickle.load(open('lgbm_model', 'rb'))

import matplotlib.pyplot as plt
print(1)
#explainerModel = shap.TreeExplainer(model)
print(2)
#shap_values = explainerModel.shap_values(df_for_model_no_target)
print(3)
#shap.summary_plot(shap_values, df_for_model_no_target)
plt.show()
print(4)

#%%

print(5)
j = 0
print(7)

#%%
#shap.force_plot(explainerModel.expected_value[j], shap_values[j], df_for_model_no_target)
#print(8)
