import numpy as np
import pandas as pd
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

def target_domain_kernel_regression(x_train, y_train, X, gamma_g=1):
    #kernel solution
    K_train = sklearn.metrics.pairwise.rbf_kernel(x_train,gamma=gamma_g)
    #c = np.dot(scipy.linalg.pinv(K_train),y_train)
    try:
        c = np.dot(np.linalg.inv(K_train), y_train)
    except:
        c = np.dot(scipy.linalg.pinv(K_train,rcond=0.001), y_train)
    K_pred = sklearn.metrics.pairwise.rbf_kernel(X,x_train,gamma=gamma_g)
    ypred = np.dot(K_pred,c)
    return ypred

def target_domain_linear_regression(x_train,y_train,X):
    #Analytic solution
    X_train = np.insert(x_train,0, values=np.ones(x_train.shape[0]), axis=1)
    w_hat = np.dot(scipy.linalg.pinv(X_train),y_train)

    X_full = np.insert(X,0, values=np.ones(X.shape[0]), axis=1)
    ypred = np.dot(X_full, w_hat)
    return ypred

if __name__ == '__main__':
    #data_path = 'C:\Users\Administrator\Desktop\Master_Study_Material\电池分选项目\Bi_KM\battery490'
    """
    selected_index = list(np.loadtxt(data_path+'\select_index.txt') - 1)
    origin_data = pd.read_excel(data_path + '\dataset_seven_cells.xlsx')
    x = origin_data.iloc[selected_index, :-1].values
    y = origin_data.iloc[selected_index, -1].values
    random_index = random.sample(range(x.shape[0]), 50)
    x_train = x[random_index]
    y_train = y[random_index]
    ypred = target_domain_linear_regression(x_train,y_train,x)
    mse = mean_squared_error(y,ypred)
    np.savetxt(data_path + '\pred_Q.txt', ypred)
    """

    x = 0.1*np.array(range(1,100)).reshape(-1,1)
    y1 = 3*x + 2 + 0.01*np.random.random()
    y2 = 2*x*x + 1 + 0.01*np.random.random()

    x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1,test_size=0.3,random_state=1)
    x2_train, x2_test, y2_train, y2_test = train_test_split(x, y2, test_size=0.3, random_state=1)

    pred_result_kernel_1 = target_domain_kernel_regression(x1_train, y1_train, x, gamma_g=5)
    pred_result_linear_1 = target_domain_linear_regression(x1_train, y1_train, x)
    mse_kernel_1 = mean_squared_error(y1, pred_result_kernel_1)
    mse_linear_1 = mean_squared_error(y1, pred_result_linear_1)

    pred_result_kernel_2 = target_domain_kernel_regression(x2_train, y2_train, x, gamma_g=5)
    pred_result_linear_2 = target_domain_linear_regression(x2_train, y2_train, x)
    mse_kernel_2 = mean_squared_error(y2, pred_result_kernel_2)
    mse_linear_2 = mean_squared_error(y2, pred_result_linear_2)

    plt.plot(x, pred_result_kernel_1, 'r+', label='kernel%f2'%mse_kernel_1)
    plt.plot(x, pred_result_linear_1, 'b+', label='linear%f2'%mse_linear_1)
    plt.plot(x, y1, 'k+', label='true')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Result on linear data')
    plt.legend()
    plt.show()

    plt.plot(x, pred_result_kernel_2, 'r+', label='kernel%f2'%mse_kernel_2)
    plt.plot(x, pred_result_linear_2, 'b+', label='linear%f2'%mse_linear_2)
    plt.plot(x, y2, 'k+', label='true')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Result on nonlinear data')
    plt.legend()
    plt.show()
    
    x = 0.1*np.array(range(1,100)).reshape(-1,1)
    y1 = 3*x + 2 + 0.01*np.random.random()
    y2 = 2*x*x + 1 + 0.01*np.random.random()

    x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1,test_size=0.3,random_state=1)
    x2_train, x2_test, y2_train, y2_test = train_test_split(x, y2, test_size=0.3, random_state=1)

    pred_result_kernel_1 = target_domain_kernel_regression(x1_train, y1_train, x, gamma_g=5)
    pred_result_linear_1 = target_domain_linear_regression(x1_train, y1_train, x)
    mse_kernel_1 = mean_squared_error(y1, pred_result_kernel_1)
    mse_linear_1 = mean_squared_error(y1, pred_result_linear_1)

    pred_result_kernel_2 = target_domain_kernel_regression(x2_train, y2_train, x, gamma_g=5)
    pred_result_linear_2 = target_domain_linear_regression(x2_train, y2_train, x)
    mse_kernel_2 = mean_squared_error(y2, pred_result_kernel_2)
    mse_linear_2 = mean_squared_error(y2, pred_result_linear_2)

    plt.plot(x, pred_result_kernel_1, 'r+', label='kernel%f2'%mse_kernel_1)
    plt.plot(x, pred_result_linear_1, 'b+', label='linear%f2'%mse_linear_1)
    plt.plot(x, y1, 'k+', label='true')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Result on linear data')
    plt.legend()
    plt.show()

    plt.plot(x, pred_result_kernel_2, 'r+', label='kernel%f2'%mse_kernel_2)
    plt.plot(x, pred_result_linear_2, 'b+', label='linear%f2'%mse_linear_2)
    plt.plot(x, y2, 'k+', label='true')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Result on nonlinear data')
    plt.legend()
    plt.show()




