# %%
import pandas as pd

# %%
df_orig = pd.read_excel('../Cap_01/default_of_credit_card_clients__courseware_version_1_21_19.xls')
df_orig

# %%
df_zero_mask = df_orig == 0
df_zero_mask

# %%
feature_zero_mask = df_zero_mask.iloc[:,1:].all(axis=1)
feature_zero_mask

# %%
sum(feature_zero_mask)

# %%
df_clean = df_orig.loc[~feature_zero_mask,:].copy()
df_clean.shape

# %%
df_clean.replace({'EDUCATION': [0, 5, 6]}, value=4, inplace=True)
df_clean.replace({'MARRIAGE': 0}, value=3, inplace=True)
df_clean['ID'].nunique()

# %%
df_clean['PAY_1'].value_counts()

# %%
missing_pay_1_mask = df_clean['PAY_1'] == 'Not available'

# %%
sum(missing_pay_1_mask)

# %%
df_missing_pay_1 = df_clean.loc[missing_pay_1_mask,:].copy()
df_missing_pay_1

# %%
df = pd.read_csv('../Cap_01/data/Chapter_1_cleaned_data.csv')
df

# %%
features_response = df.columns.to_list()

# %%
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'others',
                   'university']

# %%
features_response = [item for item in features_response if item not in items_to_remove]
features_response

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[features_response[:-1]].values,
                                                    df['default payment next month'].values,
                                                    test_size=0.2,
                                                    random_state=24)

# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
import numpy as np

np.median(X_train[:,4])

# %%
np.random.seed(seed=1)

fill_values = [0, np.random.choice(X_train[:,4], size=(3021,), replace=True)]
fill_values

# %%
fill_strategy = ['mode', 'random']

# %%
fill_values[-1]

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1,2, figsize=(8,3))
bin_edges = np.arange(-2,9)
axs[0].hist(X_train[:,4], bins=bin_edges, align='left')
axs[0].set_xticks(bin_edges)
axs[0].set_title('Non-missing values of PAY_1')
axs[1].hist(fill_values[-1], bins=bin_edges, align='left')
axs[1].set_xticks(bin_edges)
axs[1].set_title('Random selection for imputation')
plt.tight_layout()

# %%
from sklearn.model_selection import KFold

k_folds = KFold(n_splits=4, shuffle=True, random_state=1)

# %%
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=9,
                            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                            max_leaf_nodes=None, min_impurity_decrease=0.0,
                            bootstrap=True, oob_score=False, n_jobs=None,
                            random_state=4, verbose=1, warm_start=False, class_weight=None)

# %%
for counter in range(len(fill_values)):
    #Copy the data frame with missing PAY_1 and assign imputed values
    df_fill_pay_1_filled = df_missing_pay_1.copy()
    df_fill_pay_1_filled['PAY_1'] = fill_values[counter]
    
    
    #Split imputed data in to training and testing, using the same
    #80/20 split we have used for the data with non-missing PAY_1
    X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = train_test_split(
        df_fill_pay_1_filled[features_response[:-1]].values,
        df_fill_pay_1_filled['default payment next month'].values,
        test_size=0.2, random_state=24)
    
    #Concatenate the imputed data with the array of non-missing data
    X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
    y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
    
    #Use the KFolds splitter and the random forest model to get
    #4-fold cross-validation scores for both imputation methods
    imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise')
    
    test_score = imputation_compare_cv['test_score']
    print(fill_strategy[counter] + ' imputation: ' +
          'mean testing score ' + str(np.mean(test_score)) +
          ', std ' + str(np.std(test_score)))

# %%
pay_1_df = df.copy()

# %%
features_to_imputation = pay_1_df.columns.tolist()
features_to_imputation

# %%
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none', 'others',
                   'university', 'default payment next month', 'PAY_1']

# %%
features_to_imputation = [item for item in features_to_imputation if item not in items_to_remove]
features_to_imputation

# %%
X_impute_train, X_impute_test, y_impute_train, y_impute_test = \
    train_test_split(pay_1_df[features_to_imputation].values,
                     pay_1_df['PAY_1'].values,
                     test_size=0.2,
                     random_state=24)

# %%
rf_impute_params= {
    'max_depth':[3, 6, 9, 12],
    'n_estimators':[10, 50, 100, 200]}

# %%
from sklearn.model_selection import GridSearchCV

cv_rf_impute = GridSearchCV(rf, param_grid=rf_impute_params, scoring='accuracy',
                            n_jobs=-1, refit=True, cv=4, verbose=2, error_score=np.nan,
                            return_train_score=True)

# %%
cv_rf_impute.fit(X_impute_train, y_impute_train)

# %%
cv_rf_impute.best_params_

# %%
cv_rf_impute.best_score_

# %%
pay_1_value_counts = pay_1_df['PAY_1'].value_counts().sort_index()
pay_1_value_counts

# %%
pay_1_value_counts/pay_1_value_counts.sum()

# %%
y_impute_predict = cv_rf_impute.predict(X_impute_test)

# %%
from sklearn import metrics

metrics.accuracy_score(y_impute_test, y_impute_predict)

# %%
fig, axs = plt.subplots(1,2, figsize=(8,3))
axs[0].hist(y_impute_test, bins=bin_edges, align='left')
axs[0].set_xticks(bin_edges)
axs[0].set_title('Non-missing values of PAY_1')
axs[1].hist(y_impute_predict, bins=bin_edges, align='left')
axs[1].set_xticks(bin_edges)
axs[1].set_title('Model-based imputation')
plt.tight_layout()

# %%
X_impute_all = pay_1_df[features_to_imputation].values
y_impute_all = pay_1_df['PAY_1'].values

# %%
rf_impute = RandomForestClassifier(n_estimators=100, max_depth=12)

# %%
rf_impute.fit(X_impute_all, y_impute_all)

# %%
df_fill_pay_1_model = df_missing_pay_1.copy()
df_fill_pay_1_model['PAY_1'].head()

# %%
df_fill_pay_1_model['PAY_1']= rf_impute.predict(df_fill_pay_1_model[features_to_imputation].values)
df_fill_pay_1_model['PAY_1'].head()

# %%
df_fill_pay_1_model['PAY_1'].value_counts().sort_index()

# %%
X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
train_test_split(
    df_fill_pay_1_model[features_response[:-1]].values,
    df_fill_pay_1_model['default payment next month'].values,
test_size=0.2, random_state=24)

# %%
print(X_fill_pay_1_train.shape)
print(X_fill_pay_1_test.shape)
print(y_fill_pay_1_train.shape)
print(y_fill_pay_1_test.shape)

# %%
X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)

# %%
print(X_train_all.shape)
print(y_train_all.shape)

# %%
rf

# %%
imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise')

# %%
np.mean(imputation_compare_cv['test_score'])

# %%
df_fill_pay_1_model['PAY_1'] = np.zeros_like(df_fill_pay_1_model['PAY_1'].values)
df_fill_pay_1_model['PAY_1'].unique()

# %%
X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
train_test_split(
    df_fill_pay_1_model[features_response[:-1]].values,
    df_fill_pay_1_model['default payment next month'].values,
test_size=0.2, random_state=24)

# %%
X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
X_test_all = np.concatenate((X_test, X_fill_pay_1_test), axis=0)
y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
y_test_all = np.concatenate((y_test, y_fill_pay_1_test), axis=0)

# %%
print(X_train_all.shape)
print(X_test_all.shape)
print(y_train_all.shape)
print(y_test_all.shape)

# %%

imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise')

# %%
np.mean(imputation_compare_cv['test_score'])

# %%
rf.fit(X_train_all, y_train_all)

# %%
y_test_all_predict_proba = rf.predict_proba(X_test_all)

# %%
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test_all, y_test_all_predict_proba[:,1])

# %%
# Exercício
thresholds = np.linspace(0, 1, 101)

# %%
df[features_response[:-1]].columns[5]

# %%
savings_per_default = np.mean(X_test_all[:,5])
savings_per_default

# %%
cost_per_counseling = 7500

# %%
effectiveness = 0.70

# %%
n_pos_pred = np.empty_like(thresholds)
cost_of_all_counselings = np.empty_like(thresholds)
n_true_pos = np.empty_like(thresholds)
savings_of_all_counselings = np.empty_like(thresholds)

# %%
counter = 0
for threshold in thresholds:
    pos_pred = y_test_all_predict_proba[:,1]>threshold
    n_pos_pred[counter] = sum(pos_pred)
    cost_of_all_counselings[counter] = n_pos_pred[counter] * cost_per_counseling
    true_pos = pos_pred & y_test_all.astype(bool)
    n_true_pos[counter] = sum(true_pos)
    savings_of_all_counselings[counter] = n_true_pos[counter] * savings_per_default * effectiveness
    
    counter += 1

# %%
net_savings = savings_of_all_counselings - cost_of_all_counselings

# %%
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 400
plt.plot(thresholds, net_savings)
plt.xlabel('Threshold')
plt.ylabel('Net savings (NT$)')
plt.xticks(np.linspace(0,1,11))
plt.grid(True)

# %%
max_savings_ix = np.argmax(net_savings)

# %%
thresholds[max_savings_ix]

# %%
net_savings[max_savings_ix]

# %%
cost_of_defaults = sum(y_test_all) * savings_per_default
cost_of_defaults

# %%
net_savings[max_savings_ix]/cost_of_defaults

# %%
net_savings[max_savings_ix]/len(y_test_all)

# %%
plt.plot(cost_of_all_counselings/len(y_test_all), net_savings/len(y_test_all))
plt.xlabel('Upfront investment: cost of counselings per account (NT$)')
plt.ylabel('Net savings per account (NT$)')

# %%
plt.plot(thresholds, n_pos_pred/len(y_test_all))
plt.ylabel('Flag rate')
plt.xlabel('Threshold')

# %%
plt.plot(n_true_pos/sum(y_test_all), np.divide(n_true_pos, n_pos_pred))
plt.xlabel('Recall')
plt.ylabel('Precision')

# %%
plt.plot(thresholds, np.divide(n_true_pos, n_pos_pred), label='Precision')
plt.plot(thresholds, n_true_pos/sum(y_test_all), label='Recall')
plt.xlabel('Threshold')
plt.legend()

# %%
