"""
Enter a number from 1 to 5:
"""
print(__doc__)

import numpy as np
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


from sklearn.tree import DecisionTreeClassifier


# Load data
data = np.genfromtxt('year{}.csv'.format(input()), delimiter=',', missing_values='?', filling_values=0)
print(type(data))
X = data[:, :-1]
y = data[:, -1]

#Normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------------------
# Decision Tree
# Train Model
clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False).fit(X_train, y_train)

# Predict Model
y_predict_train = clf.predict(X_train)
y_predict_test = clf.predict(X_test)

# Compute Results
print('Decision Tree')
print('AUC on Train set: ',roc_auc_score(y_train, y_predict_train))
print('AUC on Test set: ',roc_auc_score(y_test, y_predict_test))
print('Accuracy on Train set: ',accuracy_score(y_train, y_predict_train))
print('Accuracy on Test set: ',accuracy_score(y_test, y_predict_test))
# -------------------------------------------------------------------------
# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Train Model
clf = RandomForestClassifier(n_estimators=20, criterion='gini', max_depth=2, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=30, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=0, verbose=0, warm_start=False, class_weight=None).fit(X_train, y_train)

# Predict Model
y_predict_train = clf.predict(X_train)
y_predict_test = clf.predict(X_test)

# Compute Results
print('\nRandom Forest')
print('AUC on Train set: ',roc_auc_score(y_train, y_predict_train))
print('AUC on Test set: ',roc_auc_score(y_test, y_predict_test))
print('Accuracy on Train set: ',accuracy_score(y_train, y_predict_train))
print('Accuracy on Test set: ',accuracy_score(y_test, y_predict_test))

# -------------------------------------------------------------------------
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Train Model
clf = GaussianNB(priors=None).fit(X_train, y_train)

# Predict Model
y_predict_train = clf.predict(X_train)
y_predict_test = clf.predict(X_test)

# Compute Results
print('\nGaussian Naive Bayes')
print('AUC on Train set: ',roc_auc_score(y_train, y_predict_train))
print('AUC on Test set: ',roc_auc_score(y_test, y_predict_test))
print('Accuracy on Train set: ',accuracy_score(y_train, y_predict_train))
print('Accuracy on Test set: ',accuracy_score(y_test, y_predict_test))

# -------------------------------------------------------------------------
# Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB

# Train Model
clf = BernoulliNB(alpha=0.5, binarize=0.0, fit_prior=False, class_prior=None).fit(X_train, y_train)

# Predict Model
y_predict_train = clf.predict(X_train)
y_predict_test = clf.predict(X_test)

# Compute Results
print('\nBernoulli Naive Bayes')
print('AUC on Train set: ',roc_auc_score(y_train, y_predict_train))
print('AUC on Test set: ',roc_auc_score(y_test, y_predict_test))
print('Accuracy on Train set: ',accuracy_score(y_train, y_predict_train))
print('Accuracy on Test set: ',accuracy_score(y_test, y_predict_test))

# -------------------------------------------------------------------------
# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# Train Model
clf = MultinomialNB(alpha=0.5, fit_prior=False, class_prior=None).fit(X_train, y_train)

# Predict Model
y_predict_train = clf.predict(X_train)
y_predict_test = clf.predict(X_test)

# Compute Results
print('\nMultinomial Naive Bayes')
print('AUC on Train set: ',roc_auc_score(y_train, y_predict_train))
print('AUC on Test set: ',roc_auc_score(y_test, y_predict_test))
print('Accuracy on Train set: ',accuracy_score(y_train, y_predict_train))
print('Accuracy on Test set: ',accuracy_score(y_test, y_predict_test))

# -------------------------------------------------------------------------
# KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Train Model
clf = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1).fit(X_train, y_train)

# Predict Model
y_predict_train = clf.predict(X_train)
y_predict_test = clf.predict(X_test)

# Compute Results
print('\nKNeighborsClassifier')
print('AUC on Train set: ',roc_auc_score(y_train, y_predict_train))
print('AUC on Test set: ',roc_auc_score(y_test, y_predict_test))
print('Accuracy on Train set: ',accuracy_score(y_train, y_predict_train))
print('Accuracy on Test set: ',accuracy_score(y_test, y_predict_test))


# -------------------------------------------------------------------------
# Support Vector Machine
from sklearn import svm

# Train Model
clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None).fit(X_train, y_train)

# Predict Model
y_predict_train = clf.predict(X_train)
y_predict_test = clf.predict(X_test)

# Compute Results
print('\nSupport Vector Machine')
print('AUC on Train set: ',roc_auc_score(y_train, y_predict_train))
print('AUC on Test set: ',roc_auc_score(y_test, y_predict_test))
print('Accuracy on Train set: ',accuracy_score(y_train, y_predict_train))
print('Accuracy on Test set: ',accuracy_score(y_test, y_predict_test))