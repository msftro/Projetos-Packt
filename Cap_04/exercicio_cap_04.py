# %%
import pandas as pd

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
features = features_response[:-1]
X = df[features].values
y = df['default payment next month'].values

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=24)

# %%
from sklearn.preprocessing import MinMaxScaler

min_max_sc = MinMaxScaler()

# %%
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='saga',
                        penalty='l1',
                        max_iter=1000)

# %%
from sklearn.pipeline import Pipeline

scale_lr_pipeline = Pipeline(steps=[
    ('scaler',min_max_sc),
    ('model',lr)
])

# %%
scale_lr_pipeline.get_params()

# %%
scale_lr_pipeline.get_params()['model__C']
scale_lr_pipeline.set_params(model__C=2)

# %%
import numpy as np

C_val_exponents = np.linspace(2,-3,6)
C_val_exponents

# %%
C_vals = np.float64(10)**C_val_exponents
C_vals

# %%
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def cross_val_C_search_pipe(k_folds, C_vals, pipeline, X, Y):
    
    n_folds = k_folds.n_splits
    cv_train_roc_auc = np.empty((n_folds, len(C_vals)))
    cv_test_roc_auc = np.empty((n_folds, len(C_vals)))
    cv_test_roc = [[]]*len(C_vals)

    for c_val_counter in range(len(C_vals)):
        #Set the C value for the model object
        pipeline.set_params(model__C = C_vals[c_val_counter])
        #Count folds for each value of C
        fold_counter = 0
        #Get training and testing indices for each fold
        for train_index, test_index in k_folds.split(X, Y):
            #Subset the features and response, for training and testing data for
            #this fold
            X_cv_train, X_cv_test = X[train_index], X[test_index]
            y_cv_train, y_cv_test = Y[train_index], Y[test_index]

            #Fit the model on the training data
            pipeline.fit(X_cv_train, y_cv_train)

            #Get the training ROC AUC
            y_cv_train_predict_proba = pipeline.predict_proba(X_cv_train)
            cv_train_roc_auc[fold_counter, c_val_counter] = \
            roc_auc_score(y_cv_train, y_cv_train_predict_proba[:,1])

            #Get the testing ROC AUC
            y_cv_test_predict_proba = pipeline.predict_proba(X_cv_test)
            cv_test_roc_auc[fold_counter, c_val_counter] = \
            roc_auc_score(y_cv_test, y_cv_test_predict_proba[:,1])

            #Testing ROC curves for each fold
            this_fold_roc = roc_curve(y_cv_test, y_cv_test_predict_proba[:,1])
            cv_test_roc[c_val_counter].append(this_fold_roc)

            #Increment the fold counter
            fold_counter += 1

        #Indicate progress
        print('Done with C = {}'.format(pipeline.get_params()['model__C']))

    return cv_train_roc_auc, cv_test_roc_auc, cv_test_roc

# %%
from sklearn.model_selection import StratifiedKFold

n_folds = 4

k_folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

cv_train_roc_auc, cv_test_roc_auc, cv_test_roc = \
cross_val_C_search_pipe(k_folds, C_vals, scale_lr_pipeline, X_train, y_train)

# %%
import matplotlib.pyplot as plt

plt.plot(C_val_exponents, np.mean(cv_train_roc_auc, axis=0), '-o',
        label='Average training score')
plt.plot(C_val_exponents, np.mean(cv_test_roc_auc, axis=0), '-x',
        label='Average testing score')
plt.ylabel('ROC AUC')
plt.xlabel('log$_{10}$(C)')
plt.legend()
plt.title('Cross validation on Case Study problem')

# %%
np.mean(cv_test_roc_auc, axis=0)

# %%
from sklearn.preprocessing import PolynomialFeatures

make_interactions = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interact = make_interactions.fit_transform(X)

# %%
X_interact.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(
X_interact, df['default payment next month'].values,
test_size=0.2, random_state=24)

# %%
print(X_train.shape)
print(X_test.shape)

# %%
cv_train_roc_auc, cv_test_roc_auc, cv_test_roc = \
cross_val_C_search_pipe(k_folds, C_vals, scale_lr_pipeline, X_train, y_train)

# %%
plt.plot(C_val_exponents, np.mean(cv_train_roc_auc, axis=0), '-o',
        label='Average training score')
plt.plot(C_val_exponents, np.mean(cv_test_roc_auc, axis=0), '-x',
        label='Average testing score')
plt.ylabel('ROC AUC')
plt.xlabel('log$_{10}$(C)')
plt.legend()
plt.title('Cross validation on Case Study problem')

# %%
np.mean(cv_test_roc_auc, axis=0)

# %%
