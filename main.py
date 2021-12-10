'''
File implementing principal-components ridge regression estimator.
The estimator penalizes the scale of the coefficient vector B in the
directions of the principal components, potentially shrinking 
components more aggressively where we have less "information".

Use data from lprostate dataset
'''

import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

'''
Estimate the beta for ridge regression
'''
def compute_beta(X_train, y_train, tau):
    inverse_mat = np.linalg.inv(X_train.T @ X_train + tau*np.identity(len(X_train.T)))
    beta = inverse_mat @ X_train.T @ y_train
    return beta.to_numpy()

'''
Compute the held-out risk given an estimated beta and test data
'''
def compute_risk(X_test, y_test, beta):
    return (np.linalg.norm((X_test @ beta - y_test))**2)/len(y_test)

'''
Calculates the out-of-sample risk given training data using the
principal-components ridge regression estimator. Evaluates the
out-of-sample risk 
'''
def get_optimal_risk(X_train, y_train, train_size, X_test, y_test):
    d = X_train.shape[1]
    # calculate an estimate for the variance using a regular LS-estimator
    b_train = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)
    sigma_hat = (1/(train_size - d))* (np.linalg.norm((X_train@b_train - y_train))**2)

    # use the SVD of the training data to calculate the optimal lambda value
    u, gamma, vh = np.linalg.svd(X_train, full_matrices=False)

    # compute lambda
    lamb = np.zeros(d)
    for j in range(len(lamb)):
        u_dot_y = (u[:,j] @ y_train)**2
        if sigma_hat >= u_dot_y:
            lamb[j] = np.inf
        else:
            val = (sigma_hat*(gamma[j]**2))/(u_dot_y - sigma_hat)
            lamb[j] = val

    # compute beta using the computed lambda values
    gamma_mat = np.diag(gamma)
    lamb_mat = np.diag(lamb)
    inverse_mat = np.linalg.inv((gamma_mat**2) + lamb_mat)
    beta_hat = vh.T @ gamma_mat @ (inverse_mat) @ u.T @ y_train

    risk_hat = compute_risk(X_test, y_test, beta_hat)
    return risk_hat

'''
Runs an 'experiment' that generates the difference in test-set risk
between the optimal ridge estimator vs. a standard ridge regression
estimate 
Input: list of penalties (tau) to use for standard ridge regression estimates
Output: list of differences
'''
def run_experiment(tau_list):

    df = pd.read_table('lprostate.dat', engine='python', header=0, index_col = 0)

    n = df.shape[0]
    train_size = int(n * 0.6)
    perm = np.random.choice(n, n, replace=False)
    df_train = df.iloc[perm[:train_size]]
    df_test = df.iloc[perm[train_size:]]

    # Create standardized training and test sets
    X_train = df_train.drop('lpsa', axis=1)
    y_train = df_train['lpsa']
    X_test = df_test.drop('lpsa', axis=1)
    y_test = df_test['lpsa']

    X_train_mean = X_train.mean()
    X_train_sd = X_train.std()
    y_train_mean = y_train.mean()

    y_train -= y_train_mean
    X_train = (X_train - X_train_mean) / X_train_sd
    y_test -= y_train_mean
    X_test = (X_test - X_train_mean) / X_train_sd

    # calculate risk using optimal penalties
    risk_hat = get_optimal_risk(X_train, y_train, train_size, X_test, y_test)

    risk_list = []
    for tau in tau_list:
        beta = compute_beta(X_train, y_train, tau)
        risk = compute_risk(X_test, y_test, beta)
        risk_list.append(risk)

    return risk_list - risk_hat

def main():

    outfile = 'optimal_vs_ridge_risk_diff.png'

    num_experiments = 25    # define number of experiments to run
    tau_list = [math.pow(10, i/10) for i in range(-10,21,1)]    # define penalities to use for standard ridge estimators to compare to
    
    risk_mat = np.zeros((num_experiments,len(tau_list)))    # define matrix to store all 
    for i in range(num_experiments):
        risk_list = run_experiment(tau_list)
        risk_mat[i] = risk_list

    # calculate the average diff. in risk for each tau
    avg_gap = np.mean(risk_mat, axis=0)

    # plot
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(tau_list, avg_gap)
    ax.set_xscale('log')
    fig.suptitle("Avg. risk diff. PC-ridge vs standard ridge")
    ax.set_xlabel('Log tau')
    ax.set_ylabel('avg. gap')
    plt.savefig(outfile)

main()