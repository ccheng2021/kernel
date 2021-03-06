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
sample=np.array([[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2]])
sample_x=np.array([[0,1],[1,2],[2,3]])
sample_x_prime=np.array([0,1,2,3])
sample_y=np.array([[1],[-1],[1]])

sample_y_trans=np.transpose(sample_y)

yy_prime=np.matmul(sample_y,sample_y_trans)
# print("yy")
# print(yy_prime)
yy_f=LA.norm(np.inner(yy_prime,yy_prime))
# print("yy_f")
# print(yy_f)


def linear_kenel(matrix):
    matrix_tran=np.transpose(matrix)
    print(matrix)
    print(matrix_tran)
    kernel_matrix=np.matmul(matrix,matrix_tran)
    print(kernel_matrix)
    # print(sk.metrics.pairwise.linear_kernel(matrix))
    return kernel_matrix
print("linear kernel")
linear_kenel(sample_x)
# linear_kenel(sample_y)
# linear_kenel(sample_x_prime)

def kernel_matrix(sample_x):
    print(sk.metrics.pairwise.pairwise_kernels(sample_x,sample_x,metric='linear'))
    return
def kernel_alignment_linear():
    m = sample_x.shape[0]
    k = sk.metrics.pairwise.pairwise_kernels(sample_x, sample_x, metric='linear')
    # yy is wrong defined here
    yy=sk.metrics.pairwise.pairwise_kernels(sample_y, sample_y, metric='linear')
    # print(k)
    # print(yy)
    inner_kyy=np.inner(k,yy)
    inner_kk=np.inner(k,k)
    print(np.inner(yy,yy))
    print("yy_a")
    print(LA.norm(np.inner(yy,yy)))
    print(inner_kk)
    alignment0=inner_kyy/(m*np.sqrt(inner_kk))#a matrix
    alignment = LA.norm(inner_kyy)/ (m * np.sqrt(LA.norm(inner_kk)))
    print(alignment)
    return alignment
# kernel_alignment_linear()

def kernel_alignment_sigmoid():
    m = sample_x.shape[0]
    k = sk.metrics.pairwise.sigmoid_kernel(sample_x, coef0=1)
    # yy is wrong defined here
    yy = sk.metrics.pairwise.sigmoid_kernel(sample_y)
    print(k)
    print(yy)

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


###################################Gaussian Kernel Matrix##############
def GaussianMatrix(X,sigma):
    row,col=X.shape
    GassMatrix=np.zeros(shape=(row,row))
    X=np.asarray(X)
    i=0
    for v_i in X:
        j=0
        for v_j in X:
            GassMatrix[i,j]=Gaussian(v_i.T,v_j.T,sigma)
            # print(GassMatrix)
            j+=1
        i+=1
    print(GassMatrix)
    return GassMatrix
def Gaussian(x,z,sigma):
    return np.exp((-(np.linalg.norm(x-z)**2))/(2*sigma**2))
GaussianMatrix(sample_x,0.5)
###########################################################################
sample=np.array([[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2]])
def optimization():
    print("linear")
    k1=sk.metrics.pairwise.pairwise_kernels(sample,sample,metric='linear')
    # print(k1)
    # kk=linear_kenel(sample)
    # print(kk)
    # for e in kk:
    #     print(e)
    #     print("eigenvec")
    #     eigen = la.eig(e)
    #     print(eigen[1])
    # print("sigmoid")
    k2=sk.metrics.pairwise.sigmoid_kernel(sample,coef0=1)
    # k1_alignment=kernel_alignment_linear()
    # k2_alignment=kernel_alignment_sigmoid()
    m=len(sample)
    print(k1)
    print(k2)
    eigen=la.eig(k1)
    print("eigen")
    print(eigen)
    k1_eigenvalue_list=eigen[0]
    k1_eigenvec_list=eigen[1]
    print("k1eigen")
    print(k1_eigenvec_list)
    print("代入公式")
    lambda1 = 1
    alignment1=0

    for v in k1_eigenvec_list:
        vv_prime=np.matmul(v,np.transpose(v))
        alignment1 = alignment1+np.square(LA.norm(np.inner(vv_prime, yy_prime)))
    print("alignment1")
    print(alignment1/(2*lambda1))
    # k1_trans_eigenvec=np.transpose(eigen[1])
    # eigen2=la.eig(k2)
    # k2_eigenvalue = eigen[0]
    # k2_eigenvec = eigen[1]
    # k2_trans_eigenvec = np.transpose(eigen[1])
    # vv_prime_1=np.matmul(k1_eigenvec,np.transpose(k1_eigenvec))
    # vv_prime_2=np.matmul(k2_eigenvec,np.transpose(k2_eigenvec))
    # print("vvprime")
    # print(vv_prime_1)
    # print(vv_prime_2)
    # print(yy_prime)
    # aligenment_1=np.square(LA.norm(np.inner(vv_prime_1,yy_prime)))
    # aligenment_2= np.square(LA.norm(np.inner(vv_prime_2,yy_prime)))
    #
    # lambda1=1
    # F_1=aligenment_1
    # F_2=aligenment_2
    # alfa_1=F_1/(2*lambda1)
    # alfa_2=F_2/(2*lambda1)
    # print("allignement")
    # print(aligenment_1)
    # print(aligenment_2)
    # print(alfa_1)
    # # w_a=np.square(LA.norm(np.inner(k1_eigenvec,sample_y)))
    # # print(w_a)
    # print("代入公式")
    # A_hat_1=((alfa_1*aligenment_1)+(alfa_2*aligenment_2))\
    #         /(np.square(yy_f)*(np.square(alfa_1)+np.square(alfa_2)))
    # print(A_hat_1)
    # print("简写")
    # A_hat_2= np.sqrt(np.square(aligenment_1)+np.square(aligenment_2)) / len(sample_y)
    # print(A_hat_2)
    return
optimization()
