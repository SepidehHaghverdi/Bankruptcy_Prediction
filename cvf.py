"""
Enter a number from 1 to 5:
"""
print(__doc__)

import numpy as np
#import matplotlib.pyplot as plt


from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
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

# -------------------------------------------------------------------------
# Decision Tree
# Train Model
clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)


print('Decision Tree')
scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
print("AUC: ")
print("mean:", scores.mean(), " std:", scores.std())

scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print("Accuracy: ")
print("mean:", scores.mean(), " std:", scores.std())

# -------------------------------------------------------------------------
# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Train Model
clf = RandomForestClassifier(n_estimators=20, criterion='gini', max_depth=2, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=30, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=0, verbose=0, warm_start=False, class_weight=None)


print('Random Forest')
scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
print("AUC: ")
print("mean:", scores.mean(), " std:", scores.std())

scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print("Accuracy: ")
print("mean:", scores.mean(), " std:", scores.std())


# -------------------------------------------------------------------------
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Train Model
clf = GaussianNB(priors=None)

print('\nGaussian Naive Bayes')
scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
print("AUC: ")
print("mean:", scores.mean(), " std:", scores.std())

scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print("Accuracy: ")
print("mean:", scores.mean(), " std:", scores.std())

# -------------------------------------------------------------------------
# Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB

# Train Model
clf = BernoulliNB(alpha=0.5, binarize=0.0, fit_prior=False, class_prior=None)

print('\nBernoulli Naive Bayes')
scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
print("AUC: ")
print("mean:", scores.mean(), " std:", scores.std())

scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print("Accuracy: ")
print("mean:", scores.mean(), " std:", scores.std())

# -------------------------------------------------------------------------
# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# Train Model
clf = MultinomialNB(alpha=0.5, fit_prior=False, class_prior=None)


print('\nMultinomial Naive Bayes')
scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
print("AUC: ")
print("mean:", scores.mean(), " std:", scores.std())

scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print("Accuracy: ")
print("mean:", scores.mean(), " std:", scores.std())


# -------------------------------------------------------------------------
# KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Train Model
clf = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)



print('\nKNeighborsClassifier')
scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
print("AUC: ")
print("mean:", scores.mean(), " std:", scores.std())

scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print("Accuracy: ")
print("mean:", scores.mean(), " std:", scores.std())

# -------------------------------------------------------------------------
# Support Vector Machine
from sklearn import svm

# Train Model
clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)

print('\nSupport Vector Machine')

scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
print("AUC: ")
print("mean:", scores.mean(), " std:", scores.std())

scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print("Accuracy: ")
print("mean:", scores.mean(), " std:", scores.std())
