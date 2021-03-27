import freqopttest.tst as tst

import data
import density
import goftest as gof
import mmd as mgof
import kernel as ker
import util
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats




# font options
font = {
    #'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 16
}

plt.rc('font', **font)
plt.rc('lines', linewidth=2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



# true p
seed = 20
d = 2
# sample
n = 400
alpha = 0.05

mean = np.zeros(d)
variance = 1


p = density.IsotropicNormal(mean, variance)
q_mean = mean.copy()
q_variance = variance
# q_mean[0] = 1

ds = data.DSIsotropicNormal(q_mean+1, q_variance)
# q_means = np.array([ [0], [0]])
# q_variances = np.array([0.01, 1])
# ds = data.DSIsoGaussianMixture(q_means, q_variances, pmix=[0.2, 0.8])

# Test
dat = ds.sample(n, seed=seed+2)
X = dat.data()
# Use median heuristic to determine the Gaussian kernel width
sig2 = util.meddistance(X, subsample=1000)**2
k = ker.KGauss(sig2)


mmd_test = mgof.QuadMMDGof(p, k, n_permute=300, alpha=alpha, seed=seed)
mmd_result = mmd_test.perform_test(dat)
mmd_result


print('Reject H0?: {0}'.format(mmd_result['h0_rejected']))

sim_stats = mmd_result['list_permuted_mmd2']
stat = mmd_result['test_stat']
unif_weights = np.ones_like(sim_stats)/float(len(sim_stats))
plt.hist(sim_stats, label='Simulated', weights=unif_weights)
plt.plot([stat, stat], [0, 0], 'r*', markersize=30, label='Stat')
plt.legend(loc='best')

