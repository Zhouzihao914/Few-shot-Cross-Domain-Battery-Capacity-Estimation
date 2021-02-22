import random
import sklearn
import scipy.io
import pandas as pd
import numpy as np
import jdot
import matplotlib.pyplot as plt
import numpy as np
from LapRLS import LapRLS
from coreg import Coreg
import scipy
import ot
from sklearn.kernel_ridge import KernelRidge
from target_self_regression import target_domain_kernel_regression
from kernel_equality import jdot_kernel_equality
from target_self_regression import target_domain_linear_regression
from cvxopt import matrix, solvers
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

#Load and preprocess the data
data_path = r'C:\Users\Administrator\Desktop\Recent_DDL\KDD\github_code'
xs = pd.read_csv(data_path + r'\xs_nasa.csv',index_col= 0).values
ys = pd.read_csv(data_path + r'\ys_nasa.csv',index_col= 0).values
xt = pd.read_csv(data_path + r'\xt_nasa.csv',index_col= 0).values
yt = pd.read_csv(data_path + r'\yt_nasa.csv',index_col= 0).values

plt.hist(yt, color='r', label='yt')
plt.xlim(1.2,2.0)
plt.title('yt')
plt.show()

plt.hist(ys, color='b', label='ys')
plt.xlim(1.2,2.0)
plt.title('ys')
plt.show()

n = len(xs)
ntest = len(xt)
idv = np.random.permutation(n)
alpha = 0.5
sample_size = 6

# Carefully choose samples, retired batteries often have relatively low capacity
# This step strongly influences the result.
# Average the results by 20 times
itermax = 15
average_times = 20
seed_list = list(range(1,average_times+1))
mse_jdot_list = []
mse_self_list = []
mse_coreg_list = []
mse_laprls_list = []
mse_basic_list = []

last_jdot_pred = []
last_self_pred = []
last_coreg_pred = []
last_laprls_pred = []

for i in range(average_times):
    random.seed(seed_list[i])
    #choose_index = np.argwhere(yt <= 0.9*yt.max())[:, 0].reshape(-1, 1)
    choose_index = random.sample(range(ntest), 40)
    ytest_index = np.array(random.sample(list(choose_index), sample_size)).flatten()
    #ytest_index = sample_index
    yt_select = yt[ytest_index]
    remain_index = list(set(range(xt.shape[0]))-set(list(ytest_index)))

    """
    Differents kinds of Semi-JDOT models
    """
    param, loss, G, ypred1 = jdot_kernel_equality(xs, ys, xt, yt, ytest_index, numIterBCD=itermax, alpha=alpha)
    mse_jdot_list.append(sklearn.metrics.mean_squared_error(ypred1, yt))

    """
    Target domain self regression model
    """
    KRR = KernelRidge(kernel='rbf', gamma=0.01, alpha=1)
    KRR.fit(xt[ytest_index], yt_select)
    ypred_self = KRR.predict(xt)

    mse_self_list.append(sklearn.metrics.mean_squared_error(ypred_self, yt))

    """
    Co-training regression model
    """
    k1 = 5
    k2 = 5
    p1 = 2
    p2 = 5
    max_iters = 10
    pool_size = 20
    verbose = False
    cr = Coreg(k1, k2, p1, p2, max_iters, pool_size)
    cr.add_data('none', xt, yt, 'battery')
    trials = 1
    cr.run_trials(ytest_index, sample_size, trials, verbose)
    pred_h1 = cr.h1.predict(xt)
    pred_h2 = cr.h2.predict(xt)
    coreg_pred = 1 / 2 * (pred_h1 + pred_h2)
    mse_coreg_list.append(sklearn.metrics.mean_squared_error(coreg_pred, yt))

    """
    lapRLS model (semi)
    """
    model = LapRLS(n_neighbors=8, bandwidth=0.001, lambda_k=0.000025, lambda_u=0.0005, solver='closed-form')
    model.fit(xt[ytest_index], xt[remain_index], yt_select)
    laprls_pred = model.predict(xt)
    mse_laprls_list.append(sklearn.metrics.mean_squared_error(laprls_pred, yt))

    """
    Basic JDOT
    """
    gamma_jdot = 1e-2
    lambd0 = 1e1
    alpha_jdot = 1e0 / 2
    model, loss = jdot.jdot_krr(xs, ys, xt, gamma_g=gamma_jdot, numIterBCD=10, alpha=alpha_jdot, lambd=lambd0,
                                ktype='rbf')
    K = sklearn.metrics.pairwise.rbf_kernel(xt, xt, gamma=gamma_jdot)
    ypred_basic = model.predict(K)
    mse_basic_list.append(sklearn.metrics.mean_squared_error(ypred_basic, yt))

result_jdot = np.average(mse_jdot_list)
result_self = np.average(mse_self_list)
result_coreg = np.average(mse_coreg_list)
result_laprls = np.average(mse_laprls_list)
result_basic = np.average(mse_basic_list)

print('mse jdot',result_jdot)
print('mse self',result_self)
print('mse coreg',result_coreg)
print('mse laprls',result_laprls)
print('mse basic',result_basic)

