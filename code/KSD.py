import numpy as np
from scipy.misc import derivative
from sklearn.datasets import fetch_species_distributions
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
from torch.autograd import Variable
import numpy as np
from scipy.misc import derivative
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sympy import *
import sklearn as sk

#####################Generate Data####################################
def gaussi_distribution():#normal distribution
    mu, sigma = 0, 0.1
    sample_gaussian_1 = np.random.normal(mu, sigma, 500)
    sample_gaussian_2 = np.random.normal(8, 0.8, 500)
    # print(sample_gaussian)
    # count, bins, ignored = plt.hist(sample_gaussian, 30, density=True)
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
    #
    #          np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
    #
    #          linewidth=2, color='r')
    #
    # plt.show()
    return sample_gaussian_1,sample_gaussian_2
# gaussi_distribution()

def exponential_distribution():
    sample_exponential= np.random.exponential(scale=2, size=(2, 3))
    sns.distplot(np.random.exponential(size=500), hist=False)
    plt.show()
    return sample_exponential
exponential_distribution()

def beta_distribution():
    sample_beta=[random.betavariate(alpha=1,beta=2) for _ in range(1,500)]
    return sample_beta
beta_distribution()

##########################Test Sample#############

def get_same_data():
    mu = -0.6
    sigma = 0.15  # 将输出数据限制到0-1之间
    same_1 = []
    for i in range(10):
        same_1.append([random.lognormvariate(mu, sigma) for _ in range(1, 10)])
    same_2 = []
    for i in range(10):
        same_2.append([random.lognormvariate(mu, sigma) for _ in range(1, 10)])
    X = torch.Tensor(same_1)
    Y = torch.Tensor(same_2)
    # print(len(X))
    X, Y = Variable(X), Variable(Y)
    print(len(X))#10
    print(len(Y[1]))#499
    return X,Y

def get_diff_data():
    mu = -0.6
    sigma = 0.15  # 将输出数据限制到0-1之间
    SAMPLE_SIZE=10
    diff_1 = []
    for i in range(10):
        diff_1.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])
    alpha = 1
    beta = 10
    diff_2 = []
    for i in range(10):
        diff_2.append([random.betavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)])

    X = torch.Tensor(diff_1)
    Y = torch.Tensor(diff_2)
    X, Y = Variable(X), Variable(Y)
    # print(len(X[1]))
    # print(len(Y))
    return X,Y
# get_diff_data()

##############################Build Kernel#################################

def linear_kenel(matrix):
    matrix_tran=np.transpose(matrix)
    kernel_matrix=np.matmul(matrix,matrix_tran)
    print(kernel_matrix)
    # print(sk.metrics.pairwise.linear_kernel(matrix))
    return kernel_matrix
# linear_kenel([[0,1],[1,2]])
# linear_kenel(sample_y)


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    print("n_sample=20")
    # print(n_samples)
    # 求矩阵的行数，即两个域的的样本总数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    print(total.size())
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    print(total0.size())
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    print(total1.size())
    # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差
    # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
    L2_distance_square =((total0 - total1) ** 2).sum(2)
    # print(((total0 - total1) ** 2).size())
    # print(L2_distance_square.size() )
    # numbers1 = torch.Tensor([[[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]],[[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]]])
    # numbers2 = torch.Tensor([[[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]],[[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]]])
    # print(numbers1.size())
    # print((numbers1 - numbers2) ** 2)
    # print(((numbers1 - numbers2) ** 2).sum(2))
    # print(torch.sum(L2_distance_square))
    # 调整高斯核函数的sigma值--
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance_square) / (n_samples ** 2 - n_samples)
    # 多核MMD
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    print(bandwidth_list)
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    print(sum(kernel_val))
    return sum(kernel_val)  # /len(kernel_val)
# X,Y=get_same_data()
# guassian_kernel(X,Y)


# kernel_size set (n,n) default
def gaussian_2d_kernel(kernel_size=3, sigma=0):
    kernel = np.zeros([kernel_size, kernel_size])
    center = kernel_size // 2

    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
            # /(np.pi * s)
    sum_val = 1 / sum_val
    return kernel * sum_val


###############################MMD##########################################
def MMD_arbitary():
    return
def MMD_RHKS(source,target,kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source_num = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    target_num = int(target.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = torch.mean(kernels[:source_num, :source_num])
    YY = torch.mean(kernels[source_num:, source_num:])
    XY = torch.mean(kernels[:source_num, source_num:])
    YX = torch.mean(kernels[source_num:, :source_num])
    loss = XX + YY - XY - YX
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算

##############################KSD#################################
fig, ax = plt.subplots(1, 1)
sample_x_1=sample_gaussian_2 = np.random.normal(8, 0.8,100)
sample_x_2=sample_gaussian_2 = np.random.normal(8, 0.8,100)
# sample_x=np.array([8.08257846,8.67351958,8.02216486,7.34998792,7.91735729,
#                     7.14284675,8.63422246,8.53355953,8.41559643,8.09294256])
sample_y=sample_gaussian_2 = np.random.normal(2,0.1,100)
# sample_y=np.array([2.19935738,2.09367552,2.11424323,1.89081297,1.88932488,1.96945423,2.03788283,
#                    1.97891364,2.04268127,1.94649662])


# print("x1")
# print(sample_x_1)
# print("x2")
# print(sample_x_2)
# print("y")
# print(sample_y)

def gaussian_pdf(x,mu,sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)

def gaissian_1d(x,sigma):
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)

def gaissian_2d(x,y,sigma):
    return np.exp(-(x** 2+y**2) / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)

def gaussian_scipy_pdf_P(x):
    def _pdf(x):
        return norm.pdf(x, 0, 2)
    return _pdf#cannot print out
# print(norm.pdf(sample_x))
# print(derivative(norm.pdf,sample_x,dx=1))

# def f(x):
#     return x**5
# print(derivative(f, x=1, dx=1e-6))
def gaussian_scipy_pdf_Q(x):
    def _pdf(x):
        return norm.pdf(x, 0.5, 8)
    return _pdf

def pdf_derivative_P(x):
    # print(derivative(norm.pdf,sample_x,dx=1))
    return derivative(norm.pdf,x,dx=1)


def RBF_gaussian_kernel(x, x_prime, sigma):
    print("RBF_gaussian_kernel")
    print(np.exp(-(LA.norm((x - x_prime))) / (2 * sigma ** 2)))
    return np.exp(-(LA.norm((x - x_prime))) / (2 * sigma ** 2))
# RBF_gaussian_kernel(X1,Y1,0.8)

sigma=2
# def f(x,x_prime):
#     return np.exp(-(np.sum((x - x_prime) ** 2)) / (2 * sigma ** 2))
x,x_prime=symbols('x,x_prime',real=True)
# print(diff(f(x,x_prime),x))


def RBF_gaussian_kernel_der_x():
    return diff(x,x)
def RBF_gaussian_kernel_der_x_prime():
    return diff(x,x_prime)

def RBF_kernel(sample_x, sample_y,sigma):
    print("RBF kernel")
    kernel = np.exp(-(LA.norm((sample_x - sample_y) ** 2)) / (2 * sigma ** 2))
    print(kernel)
    return kernel
# RBF_kernel(X1,Y1,0.8)

def RBF(X):
    kernel=sk.metrics.pairwise.rbf_kernel(X, Y=None, gamma=None)
    return kernel



def RBF_kernel_derivative_x(sample_x,sample_y,sigma):
     e=np.exp(-(LA.norm((sample_x - sample_y) ** 2)) / (2 * sigma ** 2))
     d=-(2 *sample_x - 2 * sample_y)/ (2 * sigma ** 2)
     return e*d


def RBF_kernel_derivative_y(sample_x, sample_y, sigma):
    e = np.exp(-(LA.norm((sample_x - sample_y) ** 2)) / (2 * sigma ** 2))
    d = -(- 2 * sample_y+2*sample_y) / (2 * sigma ** 2)
    x_d=e * d
    return x_d

def RBF_kernel_derivative_x_y(sample_x, sample_y, sigma):
    e = np.exp(-(LA.norm((sample_x - sample_y) ** 2)) / (2 * sigma ** 2))
    d = -(2 * sample_x - 2 * sample_y) / (2 * sigma ** 2)
    x_d=e * d
    y_d=e*d*((- 2 * sample_y) / (2 * sigma ** 2))
    return y_d


# print(norm.pdf(sample_x))
# print(derivative(norm.pdf,sample_x,dx=1))

def stain_operator(x):
    score_function=derivative(norm.pdf,x,dx=1)/norm.pdf(x)
    print("stain_operato")
    print(score_function)
    return score_function
# stain_operator(X1)
# stain_operator(sample_y)

def shuffle_x(x):
    x=np.random.shuffle(x)
    return x

def generate_U_xy():
    U_xy=np.random.multinomial(100, [1/10.]*10, size=10)
    print("U_XY")
    print(U_xy)
    return U_xy
# generate_U_xy()

X1,Y1=get_same_data()
X2,Y2=get_diff_data()
print("Same x1")
print(X1)
print(Y1)
print(len(X1[1]))#[10,49]
print("Diff Y2")
print(X2)
print(Y2)
# print(len(X1))
# print(len(X2))
# print(len(Y1))
# print(len(Y2))
print("x1x2")
print(RBF_kernel(X1,X1,sigma=0.8))
print(RBF_kernel(X1,X2,sigma=0.8))
print(np.dot(np.transpose(X1),Y1))
print(len(np.dot(np.transpose(X1),Y1)))
print("f4")
print(len(RBF_kernel_derivative_x_y(X1,Y1,sigma=0.8)[1]))#10,49
print(np.trace(RBF_kernel_derivative_x_y(X1,Y1,sigma=0.8)))

def U_q(x,x_prime):
    print("uq")
    print(stain_operator(x))
    print("RBF")
    print(RBF_kernel(x,x_prime,sigma=0.8))
    print("stop")
    f1=np.transpose(stain_operator(x))*stain_operator(x_prime)*RBF_kernel(x,x_prime,sigma=0.8)
    #operands could not be broadcast together with shapes (49,10) (10,49)
    f2=np.transpose(stain_operator(x))*RBF_kernel_derivative_y(x,x_prime,sigma=0.8)
    f3=np.transpose(RBF_kernel_derivative_y(x,x_prime,sigma=0.8))*stain_operator(x_prime)
    f4=np.trace(RBF_kernel_derivative_x_y(x,x_prime,sigma=0.8))#at least 2dim matrix
    u_q=f1+f2+f3+f4
    print(u_q)
    return u_q
# U_q(X1,Y1)
print(X1)

def S_u_hat(X):
    n=len(X)
    f1=1/(n(n-1))
    X_prime=np.shuffle(X)
    sum=np.array();
    for i in X:
        for j in X_prime:
            if i!=j:
                U_q=U_q(i,j)#u_q里面每一个值，这个思路可行吗？
        sum=sum+U_q
    return sum

def generate_weight():
    x=1
    return

def S_u_hat_star(X):
    n=len(X)
    return




def get_U_XY():

    return
def Bootstrap_Test(x):


    return