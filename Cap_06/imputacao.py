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
