import numpy as np
import math as mt
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import linear_model,gaussian_process
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
import urllib
import nD_GDA_2c as nD
from sklearn.naive_bayes import BernoulliNB # for checking purpose.......


def predict_prior_prob(y,class_value):
    prob_y =float((y==class_value).sum())/y.shape[0]
    return prob_y

def calculate_alpha(X,y):

    alpha_matrix = {}
    """print (y==np.unique(y)[0]).sum()
    print (y==np.unique(y)[1]).sum()
    print np.sum(X[y==0],axis=0)
    print np.sum(X[y==1],axis=0)"""

    alpha_matrix.update({int(np.unique(y)[0]): np.sum(X[y==np.unique(y)[0]],axis=0,dtype=np.float)/(y==np.unique(y)[0]).sum()})
    alpha_matrix.update({int(np.unique(y)[1]): np.sum(X[y==np.unique(y)[1]],axis=0,dtype=np.float)/(y==np.unique(y)[1]).sum()})

    return alpha_matrix

def membership_function(X_test,y_test,class_value,alpha,prior_probability):

    X_test = np.matrix(X_test)
    #print prior_probability
    gjx = np.zeros(X_test.shape[0])

    for i in range(X_test.shape[0]):
        gjx[i] = np.sum((np.array(X_test[i]) * np.log(alpha)) + (np.array(1-X_test[i]) * np.log(1-alpha))) + np.log(prior_probability)

    return gjx



def classify_X(X,y,fold):

    kf = KFold(X.shape[0],n_folds=fold,shuffle=True)
    current_fold = 1
    all_fold_parameter = {}
    all_para_inbuilt = {}
    for train_index, test_index in kf:

       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]


       max_gjx = {}
       alpha = calculate_alpha(X_train,y_train)
       for cl in np.unique(y):
           prior_prob = predict_prior_prob(y_train,cl)

           member_fun = membership_function(X_test,y_test,cl,alpha[cl],prior_prob)
           max_gjx.update({int(cl):member_fun})


       discri_function = nD.discriminant_function(max_gjx,y)
       y_pridct= nD.predicted_y(discri_function,max_gjx,y)

       clf = BernoulliNB()
       clf.fit(X_train,y_train)

       """print y_pridct
       print
       print clf.predict(X_test)
       exit(0)"""

       conf_matrix =nD.make_canfusion_matrix(y_test,y_pridct,y)
       #print (y_pridct==y_test).sum()

       c_m = confusion_matrix(np.array(y_test),clf.predict(X_test))  # in-built function of confusion matrix.

       all_para = nD.parameter_calculation(conf_matrix,train_file,y_test,y_pridct)
       y_pridct_inb = clf.predict(X_test)
       all_parameter = {}
       all_parameter.update({'mse':round((np.average((y_pridct_inb-y_test)**2)),4)})
       all_parameter.update({'accuracy':round(accuracy_score(y_test,y_pridct_inb),4)})
       all_parameter.update({'precision':round(precision_score(y_test,y_pridct_inb),4)})
       all_parameter.update({'recall':round(recall_score(y_test,y_pridct_inb),4)})
       all_parameter.update({'f-measure':round(f1_score(y_test,y_pridct_inb),4)})



       all_fold_parameter.update({current_fold:all_para})
       all_para_inbuilt.update({current_fold:all_parameter})

       current_fold+=1

    max_accuracy_fold = max(all_fold_parameter,key = lambda k:float(all_fold_parameter[k]['accuracy']))
    max_accuracy_fold_inbuilt = max(all_para_inbuilt,key = lambda k:float(all_para_inbuilt[k]['accuracy']))


    print
    print "-"*75
    print " "*15,"Parameter Evaluation"
    print "-"*75
    print
    nD.print_parameter(all_fold_parameter[max_accuracy_fold])
    print
    print
    print "-"*75
    print " "*15,"In Built Function Parameter"
    print "-"*75
    print
    print
    nD.print_parameter(all_para_inbuilt[max_accuracy_fold_inbuilt])







if __name__ == '__main__':


    url_file = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    raw_data = urllib.urlopen(url_file)
    train_file = np.loadtxt(raw_data,dtype = str, delimiter=',')

    single_var = np.array(train_file)
    #print single_var
    X_val = np.array(single_var[:,0:-4],dtype='float64')

    X_val[X_val!=0]=1
    X_val=np.array(X_val,dtype=np.int)

    y_val = np.array(single_var[:, -1],dtype='float64')

    #calculate_alpha(X_val,y_val)
    classify_X(X_val,y_val,10)
