import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import pandas_profiling
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from sklearn import datasets
from sklearn.feature_selection import RFE
import sklearn.metrics as metrics
from sklearn.svm import rfe
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

# Load datasets
train = pd.read_csv('/content/drive/My Drive/kdd/MSL_Dataset/Train.txt', sep=",")
test = pd.read_csv('/content/drive/My Drive/kdd/MSL_Dataset/test.txt', sep=",")

# Data preprocessing
columns = {
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"
}

train.columns = columns
test.columns = columns

# Data EDA
plt.figure(figsize=(9, 8))
sns.countplot(x="protocol_type", data=train)
plt.show()

# Model Building
train_X = train_new[cols]
train_y = train_new['attack_class']
test_X = test_new[cols]
test_y = test_new['attack_class']

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
logreg.fit(train_X, train_y)
logreg.predict(train_X)
logreg.score(train_X, train_y)

# Decision Tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
param_grid = {'max_depth': np.arange(2, 12), 'max_features': np.arange(10, 15)}
tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=10, verbose=1, n_jobs=1)
tree.fit(train_X, train_y)
tree.best_score_
tree.best_estimator_
tree.best_params_
train_pred = tree.predict(train_X)
print(metrics.classification_report(train_y, train_pred))
test_pred = tree.predict(test_X)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
param_grid_rf = {'n_estimators': [50, 60, 70, 80, 90, 100], 'max_features': [2, 3, 4, 5, 6, 7]}
gscv_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid_rf, cv=10, verbose=True, n_jobs=-1)
gscv_results = gscv_rf.fit(train_X, train_y)
gscv_results.best_params_
gscv_rf.best_score_
random_clf = RandomForestClassifier(oob_score=True, n_estimators=80, max_features=5, n_jobs=-1)
random_clf.fit(train_X, train_y)
random_test_pred = pd.DataFrame({'actual': test_y, 'predicted': random_clf.predict(test_X)})

# Support Vector Machine (SVM)
from sklearn.svm import LinearSVC
svm_clf = LinearSVC(random_state=0, tol=1e-5)
svm_clf.fit(train_X, train_y)
print(svm_clf.coef_)
print(svm_clf.intercept_)
print(svm_clf.predict(train_X))

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
model = SVC(kernel='rbf', class_weight='balanced', gamma='scale')
model.fit(train_X, train_y)

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [1, 10], 'gamma': [0.0001, 0.001]}
grid = GridSearchCV(model, param_grid)
grid.fit(train_X, train_y)
print(grid.best_params_)