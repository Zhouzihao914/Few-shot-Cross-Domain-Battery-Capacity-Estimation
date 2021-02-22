#!/usr/bin/env python
# -*- coding: utf-8 -*-



#=========================================================================================================
#================================ 0. MODULE


import numpy as np
import math
import random
from numpy import linalg

import sklearn
from sklearn import datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
import scipy.optimize as sco
import matplotlib.pyplot as plt
from itertools import cycle, islice


#=========================================================================================================
#================================ 1. ALGORITHM


class LapRLS(object):

    def __init__(self, n_neighbors, bandwidth, lambda_k, lambda_u,
                 learning_rate=None, n_iterations=None, solver='closed-form'):
        """
        Laplacian Regularized Least Square algorithm

        Parameters
        ----------
        n_neighbors : integer
            Number of neighbors to use when constructing the graph
        lambda_k : float
        lambda_u : float
        Learning_rate: float
            Learning rate of the gradient descent
        n_iterations : integer
        solver : string ('closed-form' or 'gradient-descent' or 'L-BFGS-B')
            The method to use when solving optimization problem
        """
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth
        self.lambda_k = lambda_k
        self.lambda_u = lambda_u
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.solver = solver
        

    def fit(self, X, X_no_label, Y):
        """
        Fit the model
        
        Parameters
        ----------
        X : ndarray shape (n_labeled_samples, n_features)
            Labeled data
        X_no_label : ndarray shape (n_unlabeled_samples, n_features)
            Unlabeled data
        Y : ndarray shape (n_labeled_samples,)
            Labels
        """
        # Storing parameters
        l = X.shape[0]
        u = X_no_label.shape[0]
        n = l + u
        
        # Building main matrices
        self.X = np.concatenate([X, X_no_label], axis=0)
        self.Y = np.concatenate([Y, np.zeros(u).reshape(-1,1)])
                
        # Memory optimization

        
        # Building adjacency matrix from the knn graph
        #print('Computing adjacent matrix', end='...')
        W = kneighbors_graph(self.X, self.n_neighbors, mode='connectivity')
        W = (((W + W.T) > 0) * 1)
        #print('done')

        # Computing Graph Laplacian
        #print('Computing laplacian graph', end='...')
        L = np.diag(W.sum(axis=0)) - W
        #print('done')

        # Computing K with k(i,j) = kernel(i, j)
        #print('Computing kernel matrix', end='...')
        K = rbf_kernel(self.X, gamma=self.bandwidth)
        #print('done')

        # Creating matrix J (diag with l x 1 and u x 0)
        J = np.diag(np.concatenate([np.ones(l), np.zeros(u)]))
        
        if self.solver == 'closed-form':
            
            # Computing final matrix
            #print('Computing final matrix', end='...')
            final = (J.dot(K) + self.lambda_k * l * np.identity(l + u) + ((self.lambda_u * l) / (l + u) ** 2) * L.dot(K))
            #print('done')
        
            # Solving optimization problem
            #print('Computing closed-form solution', end='...')
            self.alpha = np.linalg.inv(final).dot(self.Y)
            #print('done')
            
            # Memory optimization
            del self.Y, W, L, K, J
            
        elif self.solver == 'gradient-descent':
            """
            If solver is Gradient-descent then a learning rate and an iteration number must be provided
            """
            
            print('Performing gradient descent...')
            
            # Initializing alpha
            self.alpha = np.zeros(n)

            # Computing final matrices
            grad_part1 = -(2 / l) * K.dot(self.Y)
            grad_part2 = ((2 / l) * K.dot(J) + 2 * self.lambda_k * np.identity(l + u) + \
                        ((2 * self.lambda_u) / (l + u) ** 2) * K.dot(L)).dot(K)

            def RLS_grad(alpha):
                return np.squeeze(np.array(grad_part1 + grad_part2.dot(alpha)))
                        
            # Memory optimization
            del self.Y, W, L, K, J
        
            for i in range(self.n_iterations + 1):
                
                # Computing gradient & updating alpha
                self.alpha -= self.learning_rate * RLS_grad(self.alpha)
                
                if i % 50 == 0:
                    print("\r[%d / %d]" % (i, self.n_iterations) ,end = "")
                    
            print('\n')
        
        elif self.solver == 'L-BFGS-B':
            
            print('Performing L-BFGS-B', end='...')
            
            # Initializing alpha
            x0 = np.zeros(n)

            # Computing final matrices
            grad_part1 = -(2 / l) * K.dot(self.Y)
            grad_part2 = ((2 / l) * K.dot(J) + 2 * self.lambda_k * np.identity(l + u) + \
                        ((2 * self.lambda_u) / (l + u) ** 2) * K.dot(L)).dot(K)

            def RLS(alpha):
                return np.squeeze(np.array((1 / l) * (self.Y - J.dot(K).dot(alpha)).T.dot((self.Y - J.dot(K).dot(alpha))) \
                        + self.lambda_k * alpha.dot(K).dot(alpha) + (self.lambda_u / n ** 2) \
                        * alpha.dot(K).dot(L).dot(K).dot(alpha)))

            def RLS_grad(alpha):
                return np.squeeze(np.array(grad_part1 + grad_part2.dot(alpha)))
            
            self.alpha, _, _ = sco.fmin_l_bfgs_b(RLS, x0, RLS_grad, args=(), pgtol=1e-30, factr =1e-30)
            
            print('done')
                                    
        # Finding optimal decision boundary b using labeled data
        new_K = rbf_kernel(self.X, X, gamma=self.bandwidth)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)

        #codes below are wrote for classificaiton problems
        """
        def to_minimize(b):
            predictions = np.array((f > b) * 1)
            return - (sum(predictions == Y) / len(predictions))

        bs = np.linspace(0, 1, num=101)
        res = np.array([to_minimize(b) for b in bs])
        self.b = bs[res == np.min(res)][0]
        """
        

    def predict(self, Xtest):
        """
        Parameters
        ----------
        Xtest : ndarray shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        predictions : ndarray shape (n_samples, )
            Predicted labels for Xtest
        """

        # Computing K_new for X
        new_K = rbf_kernel(self.X, Xtest, gamma=self.bandwidth)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        #predictions = np.array((f > self.b) * 1)
        return f
    

    def accuracy(self, Xtest, Ytrue):
        """
        Parameters
        ----------
        Xtest : ndarray shape (n_samples, n_features)
            Test data
        Ytrue : ndarray shape (n_samples, )
            Test labels
        """
        predictions = self.predict(Xtest)
        mse = sklearn.metrics.mean_squared_error(predictions, Ytrue)
        print('MSE: {}'.format(mse))
        #accuracy = sum(predictions == Ytrue) / len(predictions)
        #print('Accuracy: {}%'.format(round(accuracy * 100, 2)))

if __name__ =='__main__':
    n = 200
    ntest = 200

    #random.seed(144)
    #np.random.seed(144)
    n2 = int(n / 2)
    sigma = 0.05
    xs = np.random.randn(n, 1) + 2
    xs[:n2, :] -= 4
    ys = sigma * np.random.randn(n, 1) + np.sin(xs / 2)
    #xt = np.random.randn(n, 1) + 2
    #xt[:n2, :] /= 2
    xt = xs.copy()[:ntest]
    #xt[:n2, :] -= 3
    #xt=xs

    yt = sigma * np.random.randn(ntest, 1) + np.sin(xt / 2)+0.4
    xt += 2

    fs_s = lambda x: np.sin(x / 2)
    fs_t = lambda x: np.sin((x - 2) / 2) + 0.4
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
    #random.seed(12)
    #choose_index = np.argwhere(yt <= yt.max() * 0.4)[:, 0].reshape(-1, 1)
    #selected_index = random.sample(list(choose_index.reshape(-1,1)),30)
    #selected_index = np.array(selected_index).flatten()
    selected_index = random.sample(range(ntest), 15)
    yt_selected = yt[selected_index]
    remain_index = list(set(range(xt.shape[0]))-set(list(selected_index)))

    # Laplacian Regularized Least Square   n_neighbors, bandwidth, lambda_k, lambda_u,
    #                  learning_rate=None, n_iterations=None, solver='closed-form'
    model = LapRLS(n_neighbors=10, bandwidth=0.2, lambda_k=0.00025, lambda_u=0.005, solver='closed-form')
    model.fit(xt[selected_index], xt[remain_index], yt_selected)
    pred = model.predict(xt)
    acc = model.accuracy(xt,yt)

    plt.scatter(xs, ys, c='r', edgecolors='k', label='source sample')
    plt.scatter(xt, pred, c='g', edgecolors='k', label='lapRLS pred sample')
    plt.scatter(xt[selected_index], yt[selected_index], c='b', edgecolors='k', label='selected target sample')
    plt.title('LapRLS')
    plt.show()