import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def solve():
    tr = pd.read_csv('../Dataset/reg_train.csv', index_col=['instant', 'dteday'])
    test = pd.read_csv('../Dataset/reg_test.csv', index_col=['instant', 'dteday'])
    # Get Predictors
    X = tr.drop(['cnt', 'casual', 'registered'], axis=1)
    # get response
    y = tr['cnt']
    # Put Ones in First Columns in X Like we studied
    Ones = np.ones([len(X), 1])
    X = np.column_stack([Ones, X])
    X_t = X.transpose()
    # Reformat test
    Ones = np.ones([len(test), 1])
    X_test = np.column_stack([Ones, test])
    ### compute the b` to get the RSS and MSE
    b_hat = np.dot(np.dot(np.linalg.inv(np.dot(X_t, X)), X_t), y)
    ### Compute B`xi
    b_hat_x = np.dot(X, b_hat)
    ## Compute RSS(residual some squares) with equation Sum(yi-B`xi)^2
    RSS = sum((y - b_hat_x) ** 2)
    print("Residual Sum of Squares(RSS):", RSS)
    y_hat = np.zeros([len(X_test), 1])
    for i in range(len(X_test)):
        y_hat[i] = np.dot(X_test[i], b_hat)

    # reformat and print output
    test = test.reset_index(level=1, drop=True)
    test = test.drop(
        ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum',
         'windspeed'], axis=1)
    test['cnt'] = y_hat
    # Save the data to csv format
    test.to_csv('../linear_output.csv', sep=',')


if __name__ == '__main__':
    solve()
