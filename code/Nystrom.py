
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sklearn as sk
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
import scipy.linalg as la
from numpy import linalg as LA
from scipy.linalg import svd
from sklearn.kernel_approximation import Nystroem
from sklearn import datasets, svm

sample=np.array([[1,1,2],[2,1,3],[1,1,4],[0,1,5],[2,1,6],[0,1,7]])
# sklearn
def sk_nystrom():
    clf = svm.LinearSVC()
    print(clf)
    X, y = datasets.load_digits(n_class=9, return_X_y=True)#1617 samples
    print(len(X))
    # print(len(y))
    data = X / 16.
    print(data)
    feature_map_nystroem = Nystroem(gamma=.2,random_state = 1,n_components = 300)
    # print(feature_map_nystroem)
    data_transformed = feature_map_nystroem.fit_transform(sample)
    print(clf.fit(data_transformed, y))
    print(clf.score(data_transformed, y))
    print(data_transformed)
    return
sk_nystrom()

def RBF_kernel(X):
    kernel=sk.metrics.pairwise.rbf_kernel(X, Y=None, gamma=None)
    print(kernel)
    return kernel
# RBF_kernel(sample)

def linear_kernel(X):
    kernel= sk.metrics.pairwise.linear_kernel(X)
    print(kernel)
    return kernel
# linear_kernel(sample)

def svd(X):
    kernel=linear_kernel(X)
    U, S, V = svd(kernel)
    S = np.maximum(S, 1e-12)
    normalization_ = np.dot(U / np.sqrt(S), V)
    print("s")
    print(S)
    # print("v")
    # print(v)
    # print("d")
    # print(d)
    return 0
# svd(sample)