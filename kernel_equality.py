import numpy as np
import scipy
import random
import ot
import sklearn
import pandas as pd
import pylab as pl
import numpy as np
from LapRLS import LapRLS
from sklearn.kernel_ridge import KernelRidge
from target_self_regression import target_domain_kernel_regression
from cvxopt import matrix, solvers
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def jdot_kernel_equality(X, y, Xtest, ytest, ytest_index, numIterBCD=20, alpha=1.0,
             method='emd', reg=1, gamma_g=0.01, rcond = 1e-4):
    # Initializations
    n = X.shape[0]
    ntest = Xtest.shape[0]
    wa = np.ones((n,)) / n
    wb = np.ones((ntest,)) / ntest
    # original loss
    C0 = cdist(X, Xtest, metric='sqeuclidean')
    # print np.max(C0)
    C0 = C0 / np.median(C0)
    C = alpha * C0  #+ cdist(y,ytest,metric='sqeuclidean')

    k = 0
    lamba = 1
    while (k < numIterBCD):  # and not changeLabels:
        k = k + 1
        if method == 'sinkhorn':
            G = ot.sinkhorn(wa, wb, C, reg)
        if method == 'emd':
            G = ot.emd(wa, wb, C)

        Yst = ntest * G.T.dot(y)
        #Yst[ytest_index] = ytest

        T = np.eye(ntest)
        Yt = np.zeros((ntest,1))
        Yt[ytest_index] = ytest[ytest_index]
        for i in range(ntest):
            if i not in ytest_index:
                T[i,i] = 0
        #Equality solution
        x_train = Xtest[ytest_index]
        K = sklearn.metrics.pairwise.rbf_kernel(Xtest, gamma = gamma_g)
        K_train = sklearn.metrics.pairwise.rbf_kernel(x_train, gamma = gamma_g)
        c_self = np.dot(scipy.linalg.pinv(K_train,rcond=rcond),Yt[ytest_index])

        #Matrix solution
        Left_mat = np.concatenate((2*(lamba*np.eye(ntest)+K), np.dot(T, K)),axis=0)
        Right_mat = np.concatenate((T,np.zeros((ntest,ntest))),axis=0)
        A = np.concatenate((Left_mat, Right_mat),axis=1)
        b = np.concatenate((2*Yst, Yt),axis=0)
        try:
            c_mat = np.dot(np.linalg.inv(A), b)
        except:
            c_mat = np.dot(scipy.linalg.pinv(A, rcond=rcond), b)
            #print('Use pinv')

        #ypred = g.predict(Kt)
        K_pred = sklearn.metrics.pairwise.rbf_kernel(Xtest,x_train,gamma_g)
        ypred1 = np.dot(K_pred, c_self)
        ypred2 = np.dot(K, c_mat[:Xtest.shape[0]])
        # function cost
        ypred2[ytest_index] = ytest[ytest_index]
        fcost = cdist(y, ypred2, metric='sqeuclidean')

        C = alpha * C0 + fcost

        # plot test
        """
        idt = ytest_index
        pl.scatter(X, y, c='r', edgecolors='k', label='source sample')
        pl.scatter(Xtest, ypred2, c='g', edgecolors='k', label='pred sample')
        pl.scatter(Xtest[ytest_index], ytest[ytest_index], c='b', edgecolors='k', label='selected target sample')
        for i in idt:
            ids = G[:, i].argmax()
            pl.plot([X[ids], Xtest[i]], [y[ids], ypred2[i]], 'k--')
        pl.title('Plot Test')
        pl.legend()
        pl.show()
        """

        #print('Current Loss:', np.sum(G * (fcost)))
        #print('Current W1:', W[1])
    return c_mat[:Xtest.shape[0]], np.sum(G * (fcost)), G, ypred2

if __name__ =='__main__':

    random.seed(144)
    np.random.seed(144)
    n = 200
    ntest = 200

    n2 = int(n / 2)
    sigma = 0.05
    xs = np.random.randn(n, 1) + 2
    xs[:n2, :] -= 4
    ys = sigma * np.random.randn(n, 1) + np.sin(xs / 2)

    xt = xs.copy()[:ntest]
    yt =  sigma * np.random.randn(ntest, 1) + 0.5 * np.sin(xt / 2) #+ 0.4
    xt += 2

    fs_s = lambda x: np.sin(x / 2)
    fs_t = lambda x: 0.5*np.sin((x - 2) / 2) #+ 0.4
    xvisu = np.linspace(-4, 6.5, 100)
    plt.figure(1)
    plt.clf()
    plt.subplot()
    plt.scatter(xs, ys, c='r', label='Source samples', edgecolors='k')
    plt.scatter(xt, yt, c='b', label='Target samples', edgecolors='k')
    plt.plot(xvisu, fs_s(xvisu), 'b', label='Source model')
    plt.plot(xvisu, fs_t(xvisu), 'g', label='Target model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Toy regression example')
    plt.show()

    #random pick known target label
    #random.seed(144)
    #choose_index = np.argwhere(yt <= yt.max() * 0.8)[:, 0].reshape(-1, 1)
    choose_index = np.array(random.sample(range(ntest), 50))
    selected_index = random.sample(list(choose_index.reshape(-1,1)),20)
    selected_index = np.array(selected_index).flatten()
    #selected_index = random.sample(range(ntest), 15)
    yt_selected = yt[selected_index]
    remain_index = list(set(range(xt.shape[0]))-set(list(selected_index)))
    #kernel semi jdot
    alpha = 10
    param, loss, G, ypred = jdot_kernel_equality(xs, ys, xt, yt, selected_index, alpha=alpha)
    mse_jdot = sklearn.metrics.mean_squared_error(ypred, yt)
    # Kernel Ridge Reg
    KRR = KernelRidge(kernel='rbf', gamma=0.5, alpha=1)
    KRR.fit(xt[selected_index], yt_selected)
    ypred_KRR = KRR.predict(xt)
    mse_krr = sklearn.metrics.mean_squared_error(ypred_KRR, yt)
    #laprls
    model = LapRLS(n_neighbors=10, bandwidth=0.01, lambda_k=0.00025, lambda_u=0.005, solver='closed-form')
    model.fit(xt[selected_index], xt[remain_index], yt_selected)
    laprls_pred = model.predict(xt)
    mse_lap = sklearn.metrics.mean_squared_error(laprls_pred, yt)

    print('Semi-Jdot MSE:', mse_jdot)
    print('Kernel Ridge Reg:', mse_krr)
    print('laprls Reg:', mse_lap)

    # visualization
    pl.figure(2)
    nb = max([int(0.2 * ntest), 50])
    idv = np.random.permutation(n)
    pl.scatter(xs, ys, c='r', edgecolors='k',label='source sample')
    pl.scatter(xt, ypred, c='g', edgecolors='k', label='pred sample')
    pl.scatter(xt[selected_index], yt[selected_index], c='b', edgecolors='k', label='selected target sample')
    #pl.plot(xt, ypred, 'g+')
    # pl.plot(xt[yt_selected_index], yt[yt_selected_index], 'b+')
    for i in range(nb):
        idt = G[idv[i], :].argmax()
        pl.plot([xs[idv[i]], xt[idt]], [ys[idv[i]], ypred[idt]], 'k--')

    pl.xlabel('x')
    pl.ylabel('y')
    pl.legend(loc=4)
    pl.title('semi joint OT matrices')
    pl.show()

    pl.plot(ypred, yt, 'r+')
    #pl.plot(yt_selected, yt[selected_index], 'b+')
    pl.xlabel('ypred')
    pl.ylabel('yture')
    pl.title('semi-jdot')
    pl.show()

    idt = selected_index
    pl.scatter(xs, ys, c='r', edgecolors='k', label='source sample')
    pl.scatter(xt, ypred, c='g', edgecolors='k', label='pred sample')
    pl.scatter(xt[selected_index], yt[selected_index], c='b', edgecolors='k', label='selected target sample')
    # pl.plot(xt, ypred, 'g+')
    # pl.plot(xt[yt_selected_index], yt[yt_selected_index], 'b+')
    for i in idt:
        ids = G[:, i].argmax()
        pl.plot([xs[ids], xt[i]], [ys[ids], ypred[i]], 'k--')
    pl.title('Final Plot Test')
    pl.legend()
    pl.show()


    pl.scatter(xs, ys, c='r', edgecolors='k', label='source sample')
    pl.scatter(xt, ypred_KRR, c='g', edgecolors='k', label='KRR pred sample')
    pl.scatter(xt[selected_index], yt[selected_index], c='b', edgecolors='k', label='selected target sample')
    pl.title('KRR')
    pl.show()
    # pl.plot(xt, ypred, 'g+')

    pl.scatter(xs, ys, c='r', edgecolors='k', label='source sample')
    pl.scatter(xt, laprls_pred, c='g', edgecolors='k', label='laprls pred sample')
    pl.scatter(xt[selected_index], yt[selected_index], c='b', edgecolors='k', label='selected target sample')
    pl.title('laprls')
    pl.tight_layout()
    pl.show()