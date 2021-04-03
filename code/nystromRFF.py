from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
clf = KernelRidge(alpha=1.0)
clf.fit(X, y)
KernelRidge(alpha=1.0)
print(clf.score(X,y))


# X, y = datasets.load_digits(n_class=9, return_X_y=True)
# data = X / 16.
# clf = svm.LinearSVC()
feature_map_nystroem = Nystroem(gamma=.2,random_state=1, n_components=300)
data_transformed = feature_map_nystroem.fit_transform(X)
clf.fit(data_transformed, y)
print(clf.score(data_transformed, y))


# X2= [[0, 0], [1, 1], [1, 0], [0, 1]]
# y2 = [0, 0, 1, 1]
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
feature_map_nystroem2 = Nystroem(gamma=.2,random_state=1, n_components=300)
data_transformed2 = feature_map_nystroem.fit_transform(X)
# clf = SGDClassifier(max_iter=5, tol=1e-3)
clf.fit(X_features,y)
print(clf.score(X_features, y))
clf.fit(data_transformed2,y)
print(clf.score(data_transformed2,y))