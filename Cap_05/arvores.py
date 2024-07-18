# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400

import graphviz

# %%
df = pd.read_csv('../Cap_01/data/Chapter_1_cleaned_data.csv')
df.head()

# %%
features_response = df.columns.to_list()
features_response[-5:]

# %%
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'others',
                   'university']

# %%
features_response = [item for item in features_response if item not in items_to_remove]
features_response

# %%
from sklearn.model_selection import train_test_split
from sklearn import tree

X_train, X_test, y_train, y_test = train_test_split(df[features_response[:-1]],
                                                    df['default payment next month'],
                                                    test_size=0.2,
                                                    random_state=24)

# %%
dt = tree.DecisionTreeClassifier(max_depth=2)

# %%
dt.fit(X_train, y_train)

# %%
dot_data = tree.export_graphviz(dt, out_file=None, filled=True,
                                rounded=True, feature_names=features_response[:-1],
                                proportion=True, class_names=['Not defaulted', 'Defaulted'])

# %%
graph = graphviz.Source(dot_data)
graph

# %%
sum(X_train['PAY_1']<=1.5)

# %%

