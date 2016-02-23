import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import linear_model,gaussian_process
import urllib

def make_matrix(X_train,y_train):

    z = np.ones((X_train.shape[0],1), dtype='float64')

    return np.matrix(np.c_[z,X_train]),np.matrix(y_train).T
    pass


def fit_model_theta(Z_matrix,y_matrix):


    return (((Z_matrix.T * Z_matrix)**-1) * Z_matrix.T ) * y_matrix
    pass

"""def Trainning_error(Z,theta,Y):

    train_error = (Z * theta) - Y
    return train_error"""

def Trainning_error(X,y,K_fold):
    Z,Y = make_matrix(X,y)

    mean_square_error_train_array = np.zeros(K_fold)
    for cur in range(K_fold):
        X_train,y_train,X_test,y_test = make_folds(Z,Y,cur,K_fold)
        theta = fit_model_theta(X_train,y_train)
        if(cur == 0):
            g_theta=gradient_decent(X_train,y_train)
            print
            print "Predicted theta : "
            print
            print theta
            print
            print "gradient decent Theta::"
            print
            print g_theta
            print
        #train_error = (Z * theta) - Y
        y_train_predict = predict_value(theta,X_train)
        train_error = (y_train_predict - y_train)
        mean_square_error_train_array[cur]  = np.mean(np.array(train_error)**2)
    print "Train Error:",
    print np.average(mean_square_error_train_array)
    return np.average(mean_square_error_train_array)

def Testing_error(X,y,K_fold):
    Z,Y = make_matrix(X,y)
    mean_square_error_test_array = np.zeros(K_fold)
    for cur in range(K_fold):
        X_train,y_train,X_test,y_test = make_folds(Z,Y,cur,K_fold)
        theta = fit_model_theta(X_train,y_train)
        if(cur == 0):
            g_theta=gradient_decent(X_test,y_test)
            print
            print "Predicted theta : "
            print
            print theta
            print
            print "gradient decent Theta::"
            print
            print g_theta
            print
        y_test_predict = predict_value(theta,X_test)
        y_test_error = (y_test_predict - y_test)
        mean_square_error_test_array[cur] = np.mean(np.array(y_test_error)**2)

    print "Test Error:",
    print np.average(mean_square_error_test_array)
    return np.average(mean_square_error_test_array)


def make_folds(Z,Y,cur_fold,K_fold):

    start_offset = (Z.shape[0]/K_fold)*cur_fold
    end_offset = start_offset + Z.shape[0]/K_fold

    X_test = Z[start_offset:end_offset,:]
    X_train = np.delete(Z,range(start_offset,end_offset),0)
    y_test = Y[start_offset:end_offset,:]
    y_train = np.delete(Y,range(start_offset,end_offset),0)

    return X_train,y_train,X_test,y_test


def predict_value(theta_predict,X_test):

    y_test_predict = np.zeros(len(X_test))
    for i in range(len(y_test_predict)):
            for x in range(X_test.shape[1]):
                y_test_predict[i] = y_test_predict[i]+ (theta_predict[x]*X_test[i,x])

    return y_test_predict


def gradient_decent(X,y):
    xT = X.T
    #n = np.shape(x)[1]
    iterations= 10
    learning_rate = 0.005
    theta = np.ones((X.shape[1],1),dtype='float')
    t = []
    for i in range(iterations):
        predicted = predict_value(theta,X)
        error = np.matrix(predicted).T - y
        #print error.shape
        grad = np.dot(xT, error) /X.shape[0]
        theta = theta - learning_rate*grad

    return theta


    pass





if __name__ == '__main__':


    file_no = raw_input("Enter File No: ")
    K_fold = int(raw_input("Enter No. of Fold : "))
    url_file = "http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set"+str(file_no)+".dat"
    raw_data = urllib.urlopen(url_file)
    data_path = "./"
    train_file = np.loadtxt(raw_data,dtype = str, delimiter=' ',skiprows=1)
    arr = []

    for item in train_file:
            arr.append(filter(None,item))

    single_var = np.array(arr,dtype='float64')
    #print single_var
    X_val = single_var[:,0:-1]
    y_val = single_var[:, -1]

    print
    print "TESTING PERFORMANCE :"
    Testing_error(X_val,y_val,K_fold)
    print
    print "TRAINING PERFORMANCE"
    print
    Trainning_error(X_val,y_val,K_fold)

