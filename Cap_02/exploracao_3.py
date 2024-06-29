# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 400

# %%
df = pd.read_csv('../Cap_01/data/Chapter_1_cleaned_data.csv')
df.head()

# %%
df['default payment next month'].mean()

# %%
df.groupby(['default payment next month'])['ID'].count()

# %%
from sklearn.linear_model import LogisticRegression

my_lr = LogisticRegression()
my_lr

# %%
my_new_lr = LogisticRegression(penalty='l2', 
                               dual=False, 
                               tol=0.0001, 
                               C=1.0, 
                               fit_intercept=True, 
                               intercept_scaling=1, 
                               class_weight=None, 
                               random_state=None, 
                               solver='lbfgs', 
                               max_iter=100, 
                               multi_class='deprecated', 
                               verbose=0, 
                               warm_start=False, 
                               n_jobs=None, 
                               l1_ratio=None)

# %%
my_new_lr.C = 0.1
my_new_lr.solver = 'liblinear'
my_new_lr

# %%
X = df['EDUCATION'][:10].values.reshape(-1,1)
X

# %%
y = df['default payment next month'][:10].values
y

# %%
my_new_lr.fit(X, y)

# %%
new_X = df['EDUCATION'][10:20].values.reshape(-1,1)
new_X

# %%
teste_predict = my_new_lr.predict(new_X)

# %%
valores_reais = df['default payment next month'][10:20].values

# %%
df_teste = pd.DataFrame({
    'previsto':teste_predict,
    'real':valores_reais
})

df_teste

# %%
np.random.seed(seed=1)
X = np.random.uniform(low=0.0, high=10.0, size=(1000,))
X[:10]

# %%
np.random.seed(seed=1)
slope = 0.25
intercept = -1.25
y = slope * X + np.random.normal(loc=0.0, scale=1.0, size=(1000,)) + intercept

# %%
mpl.rcParams['figure.dpi'] = 400
plt.scatter(X, y, s=1)

# %%
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.get_params()

# %%
lin_reg.fit(X.reshape(-1,1), y)
print(lin_reg.intercept_)
print(lin_reg.coef_)

# %%
y_pred = lin_reg.predict(X.reshape(-1, 1))
plt.scatter(X, y, s=1)
plt.plot(X, y_pred, 'r')

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['EDUCATION'].values.reshape(-1,1),
                                                    df['default payment next month'].values,
                                                    test_size=0.2,
                                                    random_state=24)

# %%
print('Taxa de resposta train:', y_train.mean())
print('Taxa de resposta test:', y_test.mean())

# %%
example_lr = LogisticRegression()
example_lr.get_params()

# %%
example_lr.C = 0.1
example_lr.solver = 'liblinear'
example_lr.multi_class = 'ovr'

example_lr.get_params()

# %%
example_lr.fit(X_train, y_train)

# %%
y_pred = example_lr.predict(X_test)

# %%
is_correct = y_pred == y_test
np.mean(is_correct)

# %%
from sklearn import metrics

print(example_lr.score(X_test, y_test)) # Não depende do modelo
metrics.accuracy_score(y_test, y_pred)  # Depende do modelo

# %%
P = sum(y_test)
N = sum(y_test == 0)
TP = sum((y_test == 1) & (y_pred == 1))
TPR = TP/P
FN = sum((y_test == 1) & (y_pred == 0))
FNR = FN/P
TN = sum((y_test == 0) & (y_pred == 0))
TNR = TN/N
FP = sum((y_test == 0) & (y_pred == 1))
FPR = FP/N

# Construindo a matriz de confusão manualmente usando pandas
confusion_matrix = pd.DataFrame({
    'Predicted Positive': [TP, FP],
    'Predicted Negative': [FN, TN]
}, index=['Actual Positive', 'Actual Negative'])

print(confusion_matrix)
# %%
metrics.confusion_matrix(y_test,y_pred)

# %%
y_pred_proba = example_lr.predict_proba(X_test)
y_pred_proba

# %%
prob_sum = np.sum(y_pred_proba, 1)
prob_sum

# %%
np.unique(prob_sum)

# %%
pos_proba = y_pred_proba[:,1]
pos_proba

# %%
np.histogram(pos_proba)

# %%
plt.hist(pos_proba)

# %%
mpl.rcParams['font.size'] = 12
plt.hist(pos_proba)
plt.xlabel('Predicted probability of positive class for testing data')
plt.ylabel('Number of Samples')

# %%
pos_sample_pos_proba = pos_proba[y_test==1]
neg_sample_pos_proba = pos_proba[y_test==0]

# %%
plt.hist([pos_sample_pos_proba, neg_sample_pos_proba], histtype='barstacked')
plt.legend(['Positive samples', 'Negative samples'])
plt.xlabel('Predicted probability of positive class')
plt.ylabel('Number of Samples')

# %%
fpr, tpr, thresholds = metrics.roc_curve(y_test, pos_proba)

# %%
plt.plot(fpr, tpr, '*-')
plt.plot([0, 1], [0, 1], 'r--')
plt.legend(['Logistic regression', 'Random chance'])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')

# %%
thresholds

# %%
metrics.roc_auc_score(y_test, pos_proba)

# %%
# Calcula Youden's J
youden_j = tpr - fpr
youden_j

#%%
# Encontra o índice do maior Youden's J
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

# %%
print(tpr[optimal_idx])
print(fpr[optimal_idx])

# %%
n_X = df['LIMIT_BAL'].values.reshape(-1, 1)
n_y = df['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(n_X,
                                                    n_y,
                                                    test_size=0.2,
                                                    random_state=24,
                                                    stratify=n_y)

print('Taxa de resposta train:', y_train.mean())
print('Taxa de resposta teste:', y_test.mean())

# %%
new_model = LogisticRegression()
new_model.C = 0.1
new_model.solver = 'liblinear'
new_model.get_params()

# %%
new_model.fit(X_train, y_train)
new_model

# %%
new_model_proba = new_model.predict_proba(X_test)[:,1]
new_model_proba

# %%
print(metrics.roc_auc_score(y_test, new_model_proba))

# %%
fpr, tpr, thresholds = metrics.roc_curve(y_test, new_model_proba)

# %%
plt.plot(fpr, tpr, '*-')
plt.plot([0, 1], [0, 1], 'r--')
plt.legend(['Logistic regression', 'Random chance'])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')

# %%
precision, recall, thresh = metrics.precision_recall_curve(y_test, new_model_proba)

plt.plot(recall, precision, '-x')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Logistic Regression from LIMIT_BAL')
plt.xlim([0,1])
plt.ylim([0,1])


# %%
print(metrics.auc(recall, precision))

# %%
y_train_pred_proba = new_model.predict_proba(X_train)[:,1]
print(metrics.roc_auc_score(y_train, y_train_pred_proba))

