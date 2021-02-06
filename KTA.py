print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

# import for colormaps
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sklearn as sk
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
import scipy.linalg as la
from numpy import linalg as LA

x=np.linspace(1,10, num=10)
y=np.linspace(1,10, num=10)

x, y = np.meshgrid(x, y)

z = np.exp(-0.1*x**2-0.1*y**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,y,z, cmap=cm.jet)
plt.show()


#
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn import datasets
# from sklearn.decomposition import PCA
#
# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
# y = iris.target
#
# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#
# plt.figure(2, figsize=(8, 6))
# plt.clf()
#
# # Plot the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
#             edgecolor='k')
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
#
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
#
#
# # To getter a better understanding of interaction of the dimensions
# # plot the first three PCA dimensions
# fig = plt.figure(1, figsize=(8, 6))
# ax = Axes3D(fig, elev=-150, azim=110)
# X_reduced = PCA(n_components=3).fit_transform(iris.data)
# ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
#            cmap=plt.cm.Set1, edgecolor='k', s=40)
# ax.set_title("First three PCA directions")
# ax.set_xlabel("1st eigenvector")
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel("2nd eigenvector")
# ax.w_yaxis.set_ticklabels([])
# ax.set_zlabel("3rd eigenvector")
# ax.w_zaxis.set_ticklabels([])
# plt.show()

# sample_x1=np.array([[0,1],[1,2],[2,3],[3,4],[4,5]])#x1.x2.x3.x4.x5\
# sample_y1=np.array([[1],[-1],[1],[1],[-1]])
# sample_y0=np.array([1,-1,1])

sample_x=np.array([[0,1],[1,2],[2,3]])
sample_y=np.array([[1],[-1],[1]])

def linear_kenel(matrix):
    matrix_tran=np.transpose(matrix)
    kernel_matrix=np.matmul(matrix,matrix_tran)
    print(kernel_matrix)
    # print(sk.metrics.pairwise.linear_kernel(matrix))
    return kernel_matrix
# linear_kenel([[0,1],[1,2]])
# linear_kenel(sample_y)

def kernel_matrix(sample_x):
    print(sk.metrics.pairwise.pairwise_kernels(sample_x,sample_x,metric='linear'))
    return
def kernel_alignment_linear():
    m = sample_x.shape[0]
    k = sk.metrics.pairwise.pairwise_kernels(sample_x, sample_x, metric='linear')
    yy=sk.metrics.pairwise.pairwise_kernels(sample_y, sample_y, metric='linear')
    # print(k)
    # print(yy)
    inner_kyy=np.inner(k,yy)
    inner_kk=np.inner(k,k)
    # print(np.inner(yy,yy))
    # print(LA.norm(np.inner(yy,yy)))
    # print(inner_kk)
    alignment0=inner_kyy/(m*np.sqrt(inner_kk))#a matrix
    alignment = LA.norm(inner_kyy)/ (m * np.sqrt(LA.norm(inner_kk)))
    print(alignment)
    return alignment
# kernel_alignment_linear()

def kernel_alignment_sigmoid():
    m = sample_x.shape[0]
    k = sk.metrics.pairwise.sigmoid_kernel(sample_x, coef0=1)
    yy = sk.metrics.pairwise.sigmoid_kernel(sample_y)
    # print(k)
    # print(yy)
    inner_kyy = np.inner(k, yy)
    inner_kk = np.inner(k, k)
    # print(np.inner(yy, yy))
    # print(LA.norm(np.inner(yy, yy)))
    # print(inner_kk)
    alignment0 = inner_kyy / (m * np.sqrt(inner_kk))  # a matrix
    alignment = LA.norm(inner_kyy) / (m * np.sqrt(LA.norm(inner_kk)))
    print(alignment)
    return alignment
# kernel_alignment_sigmoid()


def optimization():
    k1=sk.metrics.pairwise.pairwise_kernels(sample_x,sample_x,metric='linear')
    k2=sk.metrics.pairwise.sigmoid_kernel(sample_x,coef0=1)
    k1_alignment=kernel_alignment_linear()
    k2_alignment=kernel_alignment_sigmoid()
    eigen=la.eig(k1)
    print(eigen)
    k1_eigenvalue=eigen[0]
    k1_eigenvec=eigen[1]
    k1_trans_eigenvec=np.transpose(eigen[1])
    k2_eigenvalue = eigen[0]
    k2_eigenvec = eigen[1]
    k2_trans_eigenvec = np.transpose(eigen[1])
    print(1)
    w_a=np.square(LA.norm(np.inner(k1_eigenvec,sample_y)))
    print(w_a)
    return
optimization()
