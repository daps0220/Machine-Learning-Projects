import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import linear_model
import urllib


def Graphs_XY(X,y,value):
    plt.figure()
    plt.plot(X,y,"ro")
    plt.xlabel("X-"+value)
    plt.ylabel("y-"+value)
    plt.show()
    return

"""
Functon to predict the co -efficients.

"""

def fit_find_coefficent(X_train,y_train,degree): #Function that find theta values according to degree of polynomial

    exp_X_val = np.zeros((degree*2)+1)
    #print X_train[0]
    temp_X_train = np.zeros(X_train.shape[0])
    for i in range(len(temp_X_train)):
        temp_X_train[i] = X_train[i]
    #print temp_X_train
    #exp_X_val[0] = X_train.shape[0]
    for i in range(len(exp_X_val)):
        exp_X_val[i] = sum(X_train**i)

    exp_y_val = np.zeros(degree+1)
    exp_y_val[0] = sum(y_train)
    for i in range(1,len(exp_y_val)):
        exp_y_val[i] = sum(y_train*(temp_X_train**i))

    A = np.matrix(np.zeros((degree+1,degree+1)))
    B = np.matrix(exp_y_val)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i,j] = exp_X_val[i+j]

    #theta = np.dot((A **-1),B.T)
    theta = (A ** -1) * B.T

    #print theta
    return  theta

"""
Main Function of the program. execution start frim here.

"""

if __name__ == '__main__':


    file_no = raw_input("Enter File No : ")
    degree = int(input("Enter Polynomial Degree : "))
    K_fold = int(input("Enter cross validation Fold : "))
    data_path = "./"
    url_file = "http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set"+str(file_no)+".dat"
    raw_data = urllib.urlopen(url_file)
    train_file = np.loadtxt(raw_data,dtype = str, delimiter=' ',skiprows=1)

    arr = []

    for item in train_file:
            arr.append(filter(None,item))





    single_var = np.array(arr,dtype='float64')
    #print single_var
    X_val = single_var[:,0:-1]
    y_val = single_var[:, -1]


    """plt.plot(X_val,y_val,"ro")
    plt.xlabel("X-train")
    plt.ylabel("y-train")
    plt.show()"""




    avg_rms_array = np.zeros(degree+1)
    avg_rms_train_array = np.zeros(degree+1)


    for degree in range(1,degree+1):
        mean_square_array = np.zeros(K_fold)
        mean_square_train_array = np.zeros(K_fold)
        start_offset = single_var.shape[0] - single_var.shape[0]/K_fold
        #print  start_offset
        end_offset = single_var.shape[0]
        #print mean_square_array
        y_train_predict_all_array = []
        y_test_predict_all_array = []
        X_train__all_array = []
        X_test_all_array = []
        for k in range(K_fold):

            single_var_test = single_var[start_offset:end_offset,:]
            #single_var_train = np.delete(single_var,single_var_test,0)
            single_var_train = np.delete(single_var,range(start_offset,end_offset),0)
            #print single_var_test.shape[0]
            X_train = single_var_train[:,0:-1]
            y_train = single_var_train[:, -1]

            X_test = single_var_test[:,0:-1]
            y_test = single_var_test[:,-1]
            #print "FOLD --- " + str(k)
            theta_array = fit_find_coefficent(X_train,y_train,degree)

            # Create linear regression object
            regr = linear_model.LinearRegression()

            #Train the model using the training sets
            #regr.fit(np.vander(X_train,degree+1), y_train)

            #The coefficients
            #print('Coefficients: \n', regr.coef_)

           #print ('Predicted value : \n')
            #regr.predict(np.vander(X_test,degree+1))

            y_predict = np.zeros(len(y_test))

            for i in range(len(y_predict)):
                for d in range(0,degree+1):
                    y_predict[i] = y_predict[i] + (theta_array[d]*(X_test[i]**d))

            y_train_predict = np.zeros(len(y_train))
            for i in range(len(y_train_predict)):
                for d in range(0,degree+1):
                    y_train_predict[i] = y_train_predict[i] + (theta_array[d]*(X_train[i]**d))



            y_train_predict_all_array.append(y_train_predict)
            X_train__all_array.append(X_train)
            y_test_predict_all_array.append(y_predict)
            X_test_all_array.append(X_test)


            mean_square_error = (np.mean((y_predict - y_test)**2))
            mean_square_array[k] = mean_square_error

            mean_square_error_train = (np.mean((y_train_predict-y_train)**2))
            mean_square_train_array[k] = mean_square_error_train

            start_offset = start_offset - single_var.shape[0]/K_fold
            end_offset = end_offset - single_var.shape[0]/K_fold



        avg_rms = np.average(mean_square_array)
        avg_rms_array[degree] = avg_rms

        avg_rms_train = np.average(mean_square_train_array)
        avg_rms_train_array[degree] = avg_rms_train

        min_error_test =np.min(mean_square_array)
        min_error_train = np.min(mean_square_train_array)

        min_error_test_index = int((np.where(mean_square_array==min_error_test))[0])
        min_error_train_index = int((np.where(mean_square_train_array==min_error_train))[0])

        # Graph Plot Beginning
        X_axis_train = X_train__all_array[min_error_train_index]
        y_axis_train = y_train_predict_all_array[min_error_train_index]
        X_axis_test = X_test_all_array[min_error_test_index]
        y_axis_test = y_test_predict_all_array[min_error_test_index]
        """print
        print len(X_axis_test.tolist())
        print
        print len(y_axis_test.tolist())
        """
        #Graphs_XY(X_axis_train.tolist(),y_axis_train.tolist(),"train")           uncomment it for show a graph of train
        #Graphs_XY(X_axis_test.tolist(),y_axis_test.tolist(),"test")             uncomment it for show a gragh of test


    print "Array of MSE (test): ",
    print avg_rms_array

    print
    print "Array of MSE (train) : ",
    print avg_rms_train_array