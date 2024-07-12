# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['figure.dpi'] = 400

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
corr = df[features_response].corr()
corr.iloc[:5,:5]

# %%
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0)

# %%
X = df[features_response].iloc[:,:-1].values
y = df[features_response].iloc[:,-1].values
print(X.shape, y.shape)

# %%
from sklearn.feature_selection import f_classif

[f_stat, f_p_value] = f_classif(X, y)

# %%
f_test_df = pd.DataFrame({
    'Feature':features_response[:-1],
    'F statistic':f_stat,
    'p value':f_p_value
})

f_test_df.sort_values('p value')

# %%
from sklearn.feature_selection import SelectPercentile

selector = SelectPercentile(f_classif, percentile=20)
selector

# %%
selector.fit(X, y)

# %%
best_feature_ix = selector.get_support()
best_feature_ix

# %%
features = features_response[:-1]
best_features = [features[counter] for counter in range(len(features)) if
                 best_feature_ix[counter]]
best_features

# %%
overall_default_rate = df['default payment next month'].mean()
overall_default_rate

# %%
group_by_pay_mean_y = df.groupby('PAY_1').agg({'default payment next month':'mean'})
group_by_pay_mean_y

# %%
axes = plt.axes()
axes.axhline(overall_default_rate, color='red')
group_by_pay_mean_y.plot(marker='x', legend=False, ax=axes)
axes.set_ylabel('Proportion of credit defaults')
axes.legend(['Entire dataset', 'Groups of PAY_1'])

# %%
pos_mask = y == 1
neg_mask = y == 0

# %%
axes = plt.axes()
axes.hist(df.loc[neg_mask, 'LIMIT_BAL'], alpha=0.5, color='blue')
axes.hist(df.loc[pos_mask, 'LIMIT_BAL'], alpha=0.5, color='red')
axes.tick_params(axis='x', labelrotation=45)
axes.set_xlabel('Credit limit (NTS)')
axes.set_ylabel('Number of accounts')
axes.legend(['Not defaulted', 'Defaulted'])
axes.set_title('Credit limits by response variable')

# %%
bin_edges = list(range(0,850000,50000))
print(bin_edges[-1])

# %%
mpl.rcParams['figure.dpi'] = 400
axes = plt.axes()
axes.hist(df.loc[neg_mask, 'LIMIT_BAL'], bins=bin_edges, density=True, alpha=0.5, color='blue')
axes.hist(df.loc[pos_mask, 'LIMIT_BAL'], bins=bin_edges, density=True, alpha=0.5, color='red')
axes.tick_params(axis='x', labelrotation=45)
axes.set_xlabel('Credit limit (NTS)')
axes.set_ylabel('Number of accounts')
y_ticks = axes.get_yticks()
axes.set_yticklabels(np.round(y_ticks*50000,2))
axes.legend(['Not defaulted', 'Defaulted'])
axes.set_title('Credit limits by response variable')

# %%
X_exp = np.linspace(-4, 4, 81)
Y_exp = np.exp(X_exp)
plt.plot(X_exp, Y_exp)
plt.title('Plot of $e^X$')

#%%
X_exp = np.linspace(-4, 4, 81)
Y_exp = np.exp(-X_exp)
plt.plot(X_exp, Y_exp)
plt.title('Plot of $e^{-X}$')

# %%
def sigmoide(X):
    Y = 1 / (1 + np.exp(-X))
    return Y

# %%
X_sig = np.linspace(-7,7,141)
Y_sig = sigmoide(X_sig)
plt.plot(X_sig, Y_sig)
plt.yticks(np.linspace(0,1,11))
plt.grid()
plt.title('The sigmoid function')

# %%
p = group_by_pay_mean_y['default payment next month'].values

# %%
q = 1 - p
print(p)
print(q)

# %%
odds_ratio = p/q
odds_ratio

# %%
log_odds = np.log(odds_ratio)
log_odds

# %%
group_by_pay_mean_y.index

# %%
plt.plot(group_by_pay_mean_y.index, log_odds, '-x')
plt.ylabel('Log odds of default')
plt.xlabel('Values of PAY_1')

# %%
np.random.seed(seed=6)
X_1_pos = np.random.uniform(low=1, high=7, size=(20,1))
print(X_1_pos[:3])
X_1_neg = np.random.uniform(low=3, high=10, size=(20,1))
print(X_1_neg[:3])
X_2_pos = np.random.uniform(low=1, high=7, size=(20,1))
print(X_2_pos[:3])
X_2_neg = np.random.uniform(low=3, high=10, size=(20,1))
print(X_2_neg[:3])

# %%
plt.scatter(X_1_pos, X_2_pos, color='red', marker='x')
plt.scatter(X_1_neg, X_2_neg, color='blue', marker='x')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.legend(['Positive class', 'Negative class'])

# %%
X = np.block([[X_1_pos, X_2_pos], [X_1_neg, X_2_neg]])
print(X.shape)
print(X[:3])

# %%
y = np.vstack((np.ones((20,1)), np.zeros((20,1)))).reshape(40,)
print(y[:5])
print(y[-5:])

# %%
from sklearn.linear_model import LogisticRegression

example_lr = LogisticRegression(solver='liblinear')
example_lr

# %%
example_lr.fit(X, y)

# %%
y_pred = example_lr.predict(X)
positive_indices = [counter for counter in range(len(y_pred)) if y_pred[counter]==1]
negative_indices = [counter for counter in range(len(y_pred)) if y_pred[counter]==0]
positive_indices

# %%
plt.scatter(X_1_pos, X_2_pos, color='red', marker='x')
plt.scatter(X_1_neg, X_2_neg, color='blue', marker='x')
plt.scatter(X[positive_indices, 0], X[positive_indices, 1], s=150, marker='o',
            edgecolors='red', facecolors='none')
plt.scatter(X[negative_indices, 0], X[negative_indices, 1], s=150, marker='o',
            edgecolors='blue', facecolors='none')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.legend(['Positive class', 'Negative class',
            'Positive predictions', 'Negative predictions'])

# %%
theta_1 = example_lr.coef_[0][0]
theta_2 = example_lr.coef_[0][1]
print(theta_1, theta_2)

# %%
theta_0 = example_lr.intercept_

# %%
X_1_decision_boundary = np.array([0, 10])
X_2_decision_boundary = -(theta_1/theta_2)*X_1_decision_boundary -(theta_0/theta_2)

# %%
plt.scatter(X_1_pos, X_2_pos, color='red', marker='x', label='Positive class')
plt.scatter(X_1_neg, X_2_neg, color='blue', marker='x', label='Negative class')
plt.scatter(X[positive_indices, 0], X[positive_indices, 1], s=150, marker='o',
            edgecolors='red', facecolors='none', label='Positive predictions')
plt.scatter(X[negative_indices, 0], X[negative_indices, 1], s=150, marker='o',
            edgecolors='blue', facecolors='none', label='Negative predictions')
dec = plt.plot(X_1_decision_boundary, X_2_decision_boundary, 'k-',
               label='Decidion boundary')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.legend(loc=[0.25, 1.05])

# %%
from sklearn import metrics
from sklearn.model_selection import train_test_split

# %%
X = df[['PAY_1', 'LIMIT_BAL']]
y = df['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=24)

# %%
from sklearn import linear_model

lr = linear_model.LogisticRegression(solver='liblinear')
lr

# %%
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred_proba = lr.predict_proba(X_test)

# %%
ones_and_features = np.hstack([np.ones((X_test.shape[0],1)), X_test])

# %%
intercept_and_coefs = np.concatenate([lr.intercept_.reshape(1,1), lr.coef_], axis=1)

# %%
X_lin_comb = np.dot(intercept_and_coefs, np.transpose(ones_and_features))

# %%
y_pred_proba_manual = sigmoide(X_lin_comb)

# %%
y_pred_manual = y_pred_proba_manual >= 0.5

# %%
np.array_equal(y_pred.reshape(1,-1), y_pred_manual)

# %%
metrics.roc_auc_score(y_test, y_pred_proba_manual.reshape(y_pred_proba_manual.shape[1],))

# %%
metrics.roc_auc_score(y_test, y_pred_proba[:,1])

# %%
