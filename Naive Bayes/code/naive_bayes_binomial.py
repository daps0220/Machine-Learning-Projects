import numpy as np
import math as mt
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
import urllib
import nD_GDA_2c as nD
from sklearn.naive_bayes import MultinomialNB # for checking purpose.......
import gmpy as gm


def predict_prior_prob(y,class_value):
    prob_y =float((y==class_value).sum())/y.shape[0]
    return prob_y

def calculate_alpha(X,y,total_X_val):

    epsilon =  1
    alpha_matrix = {}

    #alpha_matrix.update({int(np.unique(y)[0]): (np.sum(X[y==np.unique(y)[0]],axis=0,dtype=np.float)+epsilon)/((total_X_val[y==np.unique(y)[0]]).sum()+len(np.unique(y))*epsilon)})
    #alpha_matrix.update({int(np.unique(y)[1]): (np.sum(X[y==np.unique(y)[1]],axis=0,dtype=np.float)+epsilon)/((total_X_val[y==np.unique(y)[1]]).sum()+len(np.unique(y))*epsilon)})

    alpha_matrix.update({int(np.unique(y)[0]): (np.sum(X[y==np.unique(y)[0]],axis=0,dtype=np.float))/((total_X_val[y==np.unique(y)[0]]).sum())})
    alpha_matrix.update({int(np.unique(y)[1]): (np.sum(X[y==np.unique(y)[1]],axis=0,dtype=np.float))/((total_X_val[y==np.unique(y)[1]]).sum())})

    #print alpha_matrix

    return alpha_matrix

def membership_function(X_test,y_test,class_value,alpha,prior_probability,total_X_val):

    X_test = np.matrix(X_test)
    #print prior_probability
    #gjx = np.zeros(X_test.shape[0])

    gjx =[]


    for i in range(X_test.shape[0]):

        for j in range(X_test.shape[1]):
            sum_gj =0.0
            combination = gm.comb(int(total_X_val[i]),int(X_test[i,j]))
            if(combination!=0 and alpha[j]!=0 and (1-alpha[j])!=0):
                sum_gj+= (mt.log(combination)) + (X_test[i,j] * mt.log(alpha[j])) + ((total_X_val[i]-X_test[i,j]) * mt.log(1-alpha[j]))
            else:
                sum_gj+= 0

        sum_gj+= mt.log(abs(prior_probability))
        """j=1
        val1 = [(mt.log(gm.comb(int(total_X_val[j]),int(X_test[j,i])))) for i in range(X_test.shape[1])]
        val2 = sum([(X_test[j,i] * mt.log(alpha[i])) for i in  range(X_test.shape[1])])
        val3 = sum([((total_X_val[j] - X_test[j,i]) * mt.log(1-alpha[i])) for i in range(X_test.shape[1])])


        total = val1 + val2 + val3 + mt.log(prior_probability)
        print total_X_val[j]+1
        print X_test[j,18]
        print mt.log(gm.comb(1183,347))
        print "Val 1 :",val1
        print "Val 2 :",val2
        print "Val 3 :",val3
        print "Total :",total
        exit(0)"""
        gjx.append(sum_gj)







    return np.array(gjx)

def classify_X(X,y,fold):

    kf = KFold(X.shape[0],n_folds=fold,shuffle=True)
    current_fold = 1
    all_fold_parameter = {}
    all_para_inbuilt = {}
    for train_index, test_index in kf:

       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]


       max_gjx = {}
       total_X_val_train = np.sum(X_train,axis=1)
       total_X_val_test = np.sum(X_test,axis=1)
       alpha = calculate_alpha(X_train,y_train,total_X_val_train)

       for cl in np.unique(y):

           prior_prob = predict_prior_prob(y_train,cl)

           member_fun = membership_function(X_test,y_test,cl,alpha[cl],prior_prob,total_X_val_test)
           max_gjx.update({int(cl):member_fun})

       #print max_gjx
       #exit(0)
       discri_function = nD.discriminant_function(max_gjx,y)
       y_pridct= nD.predicted_y(discri_function,max_gjx,y)

       clf = MultinomialNB()
       clf.fit(X_train,y_train)


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

    document_length = 100

    X_val = np.array(single_var[:,:-4],dtype='float64')
    X_val = X_val*document_length

    X_val = np.array(X_val,dtype=np.int)

    #total_X_val = np.sum(X_val,axis=1)

    y_val = np.array(single_var[:, -1],dtype=np.int)

    classify_X(X_val,y_val,10)





