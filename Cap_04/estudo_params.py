# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
X_poly = np.linspace(-3, 5, 81)
print(X_poly[:5], '...', X_poly[-5:])

# %%
def cost_function(X):
    return X * (X-2)

y_poly = cost_function(X_poly)
plt.plot(X_poly, y_poly)
plt.xlabel('Parameter value')
plt.ylabel('Cost Function')
plt.title('Error surface')

# %%
def gradient(X):
    return (2*X) - 2

x_start = 4.5
learning_rate = 0.5
x_next = x_start - gradient(x_start)*learning_rate
x_next

# %%
plt.plot(X_poly, y_poly)
plt.plot([x_start, x_next],
         [cost_function(x_start), cost_function(x_next)],
         '-o')
plt.legend(['Error surface', 'Gradient desccent path'])
plt.xlabel('Parameter value')
plt.ylabel('Cost Function')
plt.title('Error surface')

# %%
iterations = 15
x_path = np.empty(iterations,)
x_path[0] = x_start
for iteration_count in range(1, iterations):
    derivative = gradient(x_path[iteration_count-1])
    x_path[iteration_count] = x_path[iteration_count-1] - (derivative*learning_rate)
x_path

# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# %%
X_synthetic, y_synthetic = make_classification(n_samples=1000,
                                                 n_features=200,
                                                 n_informative=3,
                                                 n_redundant=10,
                                                 n_repeated=0,
                                                 n_classes=2,
                                                 n_clusters_per_class=2,
                                                 weights=None,
                                                 flip_y=0.01,
                                                 class_sep=0.8,
                                                 hypercube=True,
                                                 shift=0.0,
                                                 scale=1.0,
                                                 shuffle=True,
                                                 random_state=24)

# %%
print(X_synthetic.shape, y_synthetic.shape)

# %%
for plot_index in range(4):
    plt.subplot(2,2,plot_index+1)
    plt.hist(X_synthetic[:,plot_index])
    plt.title('Histogram for feature {}'.format(plot_index+1))
plt.tight_layout()

# %%
X_syn_train, X_syn_test, y_syn_train, y_syn_test = train_test_split(
    X_synthetic, y_synthetic,
    test_size=0.2, random_state=24
)

lr_syn = LogisticRegression(solver='liblinear',
                            penalty='l1',
                            C=10**-1.5,
                            random_state=1)

lr_syn.fit(X_syn_train, y_syn_train)

# %%
y_syn_train_predict_proba = lr_syn.predict_proba(X_syn_train)
roc_auc_score(y_syn_train, y_syn_train_predict_proba[:,1])

# %%
y_syn_test_predict_proba = lr_syn.predict_proba(X_syn_test)
roc_auc_score(y_syn_test, y_syn_test_predict_proba[:,1])

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

n_folds = 4

k_folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

# %%
C_val_exponents = np.linspace(3,-3,13)
C_val_exponents

# %%
C_vals= np.float64(10)**C_val_exponents
C_vals

# %%
from sklearn.metrics import roc_curve

def cross_val_C_search(k_folds, C_vals, model, X, Y):

    n_folds = k_folds.n_splits
    cv_train_roc_auc = np.empty((n_folds, len(C_vals)))
    cv_test_roc_auc = np.empty((n_folds, len(C_vals)))
    cv_test_roc = [[]]*len(C_vals)

    for c_val_counter in range(len(C_vals)):
        #Set the C value for the model object
        model.C = C_vals[c_val_counter]
        #Count folds for each value of C
        fold_counter = 0
        #Get training and testing indices for each fold
        for train_index, test_index in k_folds.split(X, Y):
            #Subset the features and response, for training and testing data for
            #this fold
            X_cv_train, X_cv_test = X[train_index], X[test_index]
            y_cv_train, y_cv_test = Y[train_index], Y[test_index]

            #Fit the model on the training data
            model.fit(X_cv_train, y_cv_train)

            #Get the training ROC AUC
            y_cv_train_predict_proba = model.predict_proba(X_cv_train)
            cv_train_roc_auc[fold_counter, c_val_counter] = \
            roc_auc_score(y_cv_train, y_cv_train_predict_proba[:,1])

            #Get the testing ROC AUC
            y_cv_test_predict_proba = model.predict_proba(X_cv_test)
            cv_test_roc_auc[fold_counter, c_val_counter] = \
            roc_auc_score(y_cv_test, y_cv_test_predict_proba[:,1])

            #Testing ROC curves for each fold
            this_fold_roc = roc_curve(y_cv_test, y_cv_test_predict_proba[:,1])
            cv_test_roc[c_val_counter].append(this_fold_roc)

            #Increment the fold counter
            fold_counter += 1

        #Indicate progress
        print('Done with C = {}'.format(lr_syn.C))

    return cv_train_roc_auc, cv_test_roc_auc, cv_test_roc

# %%

cv_train_roc_auc, cv_test_roc_auc, cv_test_roc = \
cross_val_C_search(k_folds, C_vals, lr_syn, X_syn_train, y_syn_train)

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

cmap = plt.get_cmap('plasma')
norm = Normalize(vmin=0, vmax=n_folds-1)
scalar_map = ScalarMappable(norm=norm, cmap=cmap)

for this_fold in range(n_folds):
    color_val = scalar_map.to_rgba(this_fold)
    plt.plot(C_val_exponents, cv_train_roc_auc[this_fold], '-o',
             color=color_val, label=f'Training fold {this_fold+1}')
    plt.plot(C_val_exponents, cv_test_roc_auc[this_fold], '-x',
             color=color_val, label=f'Testing fold {this_fold+1}')
plt.ylabel('ROC AUC')
plt.xlabel('log$_{10}$(C)')
plt.legend(loc = [1.1, 0.2])
plt.title('Cross validation scores for each fold')

# %%
plt.plot(C_val_exponents, np.mean(cv_train_roc_auc, axis=0), '-o',
        label='Average training score')
plt.plot(C_val_exponents, np.mean(cv_test_roc_auc, axis=0), '-x',
        label='Average testing score')
plt.ylabel('ROC AUC')
plt.xlabel('log$_{10}$(C)')
plt.legend()
plt.title('Cross validation scores averaged over all folds')

# %%
best_C_val_bool = C_val_exponents == -1.5
best_C_val_bool.astype(int)

# %%
best_C_val_ix = np.nonzero(best_C_val_bool.astype(int))
best_C_val_ix[0][0]

# %%
for this_fold in range(k_folds.n_splits):
    fpr = cv_test_roc[best_C_val_ix[0][0]][this_fold][0]
    tpr = cv_test_roc[best_C_val_ix[0][0]][this_fold][1]
    plt.plot(fpr, tpr, label='Fold {}'.format(this_fold+1))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curves for each fold at C = $10^{-1.5}$')
plt.legend()

# %%
from sklearn.preprocessing import MinMaxScaler

min_max_sc = MinMaxScaler()

# %%
from sklearn.pipeline import Pipeline

lr = LogisticRegression()

scale_lr_pipeline = Pipeline(steps=[
    ('scaler', min_max_sc),
    ('model', lr)
])

# %%
from sklearn.preprocessing import PolynomialFeatures

make_interactions = PolynomialFeatures(degree=2,
                                       interaction_only=True,
                                       include_bias=False)

# %%
