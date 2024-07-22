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
pm0 = np.linspace(0.01, 0.99, 99)

# %%
pm1 = 1 - pm0
misclassification_rate = np.minimum(pm0,pm1)

# %%
mpl.rcParams['figure.dpi'] = 400
plt.plot(pm0, misclassification_rate, label='Missclassification rate')
plt.xlabel('$p_{m0}$')
plt.legend()

# %%
gini = (pm0*(1-pm0)+ (pm1*(1-pm1)))

# %%
cross_ent = -1*((pm0*np.log(pm0)) + (pm1*np.log(pm1)))

# %%
plt.plot(pm0, gini, label='Gini impurity')
plt.plot(pm0, cross_ent, label='Cross entropy')
plt.plot(pm0, misclassification_rate, label='Missclassification rate')
plt.xlabel('$p_{m0}$')
plt.legend()

# %%
from sklearn.model_selection import GridSearchCV

params = {'max_depth':[1, 2, 4, 6, 8, 10, 12]}

cv = GridSearchCV(dt,param_grid=params, scoring='roc_auc',
                  n_jobs=None, refit=True, cv=4,
                  verbose=1, error_score=np.nan,
                  return_train_score=True)

# %%
cv.fit(X_train, y_train)

# %%
cv_results_df = pd.DataFrame(cv.cv_results_)
cv_results_df.sort_values(by='rank_test_score', ascending=True)

# %%
ax = plt.axes()
ax.errorbar(cv_results_df['param_max_depth'],
            cv_results_df['mean_train_score'],
            yerr=cv_results_df['std_train_score'],
            label='Mean $\\pm$ 1 SD training scores')
ax.errorbar(cv_results_df['param_max_depth'],
            cv_results_df['mean_test_score'],
            yerr=cv_results_df['std_test_score'],
            label='Mean $\\pm$ 1 SD testing scores')
ax.legend()
plt.xlabel('max_depth')
plt.ylabel('ROC AUC')

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=3,
                            min_samples_split=2, min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_leaf_nodes=None, min_impurity_decrease=0.0,
                            bootstrap=True, oob_score=False, n_jobs=None,
                            random_state=4, verbose=0, warm_start=False,
                            class_weight=None)

# %%
rf_params = {'n_estimators':list(range(10,110,10))}

# %%
cv_rf_ex = GridSearchCV(rf, param_grid=rf_params, scoring='roc_auc',
                  n_jobs=None, refit=True, cv=4,
                  verbose=1, error_score=np.nan,
                  return_train_score=True)

# %%
cv_rf_ex.fit(X_train, y_train)

# %%
cv_rf_ex_results_df = pd.DataFrame(cv_rf_ex.cv_results_)
cv_rf_ex_results_df.sort_values(by='rank_test_score', ascending=True)

# %%
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
axs[0].plot(cv_rf_ex_results_df['param_n_estimators'],
            cv_rf_ex_results_df['mean_fit_time'],
            '-o')
axs[0].set_xlabel('Number of trees')
axs[0].set_ylabel('Mean fit time (seconds)')
axs[1].errorbar(cv_rf_ex_results_df['param_n_estimators'],
                cv_rf_ex_results_df['mean_test_score'],
                yerr=cv_rf_ex_results_df['std_test_score'])
axs[1].set_xlabel('Number of trees')
axs[1].set_ylabel('Mean testing ROC AUC $\pm$ 1 SD ')
plt.tight_layout()

# %%
feat_imp_df = pd.DataFrame({
    'Feature name':features_response[:-1],
    'Importance':cv_rf_ex.best_estimator_.feature_importances_
})
feat_imp_df.sort_values('Importance', ascending=False)

# %%
xx_example, yy_example = np.meshgrid(range(5), range(5))
print(xx_example)
print(yy_example)

# %%
z_example = np.arange(1,17).reshape(4,4)
z_example

# %%
ax = plt.axes()
pcolor_ex = ax.pcolormesh(xx_example, yy_example, z_example, cmap=plt.cm.jet)
plt.colorbar(pcolor_ex, label='Color scale')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')

# %%
#Exercício

rf_params = {
    'max_depth':[3, 6, 9, 12],
    'n_estimators':[10, 50, 100, 200]
}

# %%
grid = GridSearchCV(rf, param_grid=rf_params, scoring='roc_auc',
                  n_jobs=None, refit=True, cv=4,
                  verbose=2, error_score=np.nan,
                  return_train_score=True)

# %%
grid.fit(X_train, y_train)

# %%
grid_results = pd.DataFrame(grid.cv_results_)
grid_results.sort_values('rank_test_score', ascending=True).head()

# %%
grid_results[grid_results['rank_test_score'] == 1].T

# %%
xx_rf, yy_rf = np.meshgrid(range(5), range(5))

# %%
cm_rf = plt.cm.jet

# %%
ax_rf = plt.axes()
pcolor_graph = ax_rf.pcolormesh(xx_rf, yy_rf, grid_results['mean_test_score'].values.reshape((4,4)), cmap=cm_rf)
plt.colorbar(pcolor_graph, label='Average testing ROC AUC')
ax_rf.set_aspect('equal')
ax_rf.set_xticks([0.5, 1.5, 2.5, 3.5])
ax_rf.set_yticks([0.5, 1.5, 2.5, 3.5])
ax_rf.set_xticklabels([str(tick_label) for tick_label in rf_params['n_estimators']])
ax_rf.set_yticklabels([str(tick_label) for tick_label in rf_params['max_depth']])
ax_rf.set_xlabel('Number of trees')
ax_rf.set_ylabel('Maximum depth')

# %%
grid.best_params_

# %%
feat_imp_df_act = pd.DataFrame({
    'Feature name':features_response[:-1],
    'Importance':grid.best_estimator_.feature_importances_
})

# %%
feat_imp_df_act.sort_values('Importance', ascending=False)

# %%
