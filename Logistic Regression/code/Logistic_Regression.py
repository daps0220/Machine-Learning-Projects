from sklearn.datasets import fetch_mldata
import numpy as np
import os
from pylab import *
import math as mt
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,f1_score,recall_score,precision_recall_curve,average_precision_score


def sigmoid_x(X_train,theta):
    #print (np.asarray(np.matrix(theta)*np.matrix(X_train.T)))
    #print (1/(1+np.exp((np.asarray(np.matrix(theta)*np.matrix(X_train.T))))))[0]
    #exit(0)
    #for i in :
    #print np.matrix(theta).shape
    #print X_train.T.shape
    return  (1/(1+np.exp((np.asarray(-(np.matrix(theta)*np.matrix(X_train).T))))))[0]

def softmax(X_train,theta_all_dict,cl,iter):


    numerator = np.exp((np.asarray(np.matrix(theta_all_dict[cl])*np.matrix(X_train).T)))[0]

    sum_all_theta_class = np.zeros((X_train.shape[0]),dtype=np.float)

    for key in theta_all_dict:
        sum_all_theta_class+=np.exp(np.asarray(np.matrix(theta_all_dict[key])*np.matrix(X_train).T))[0]

        """if(cl==1 and iter==2):
            print ">>>>>>>>>>>>>>>>  "
            print theta_all_dict[key]
            print X_train
            print ">>>>>>>>>>>>>>>>>"""



    return (numerator/sum_all_theta_class)

def graident_theta(X_train,y_train):

    class_theta = {}
    learning_rate = 0.0001

    if(len(np.unique(y_train))>2):

        theta_all_class = {}
        unique_y=np.unique(y_train)


        #print np.unique(y_train)
        for cl in unique_y:
            iter = 1
            X_train_class=X_train
            y_train_class=y_train
            y_train_class = y_train_class.reshape(y_train_class.shape[0],1)
            #print X_train_class.shape
            theta = np.zeros((X_train_class.shape[1]),dtype=np.float)
            #theta[0] =1
            theta_temp_class = {}
            for c in unique_y:
                theta_temp_class.update({int(unique_y[c]):theta})

            while(iter<=10):

                h_theta = softmax(X_train_class,theta_temp_class,cl,iter)
                h_theta = h_theta.reshape(y_train_class.shape[0],1)
                #print (h_theta[y_train==cl]-y_train_class[y_train==cl])*X_train_class[y_train==cl]
                #print y_train_class[y_train==cl]
                #print np.unique(y_train_class)
                #(y_train_class[y_train_class==cl])
                #print X_train_class[y_train==2]
                #print (y_train_class[y_train==cl]).shape
                #print (h_theta[y_train==cl]).shape
                #exit(0)

                value_1=(h_theta[y_train==cl]-y_train_class[y_train==cl])*X_train_class[y_train==cl]
                value_2 = np.sum(value_1,axis=0)
                value_3 = learning_rate * value_2

                theta_temp_class[cl] = theta_temp_class[cl] - value_3#(learning_rate*(np.sum(((h_theta[y_train==cl]-y_train_class[y_train==cl])*X_train_class[X_train==cl]),axis=0)))

                iter+=1
            #print theta_temp_class[cl]
            theta_all_class.update({int(cl):np.matrix(theta_temp_class[cl])})

        #print theta_all_class
        #exit(0)


        return theta_all_class
    else :
         X_train_class=X_train
         y_train_class=y_train
         y_train_class = y_train_class.reshape(y_train_class.shape[0],1)
         #print X_train_class.shape
         theta = np.zeros((X_train_class.shape[1]),dtype=np.float)
         #theta[0] =1
         iter = 1
         #print np.unique(y_train)
         while(iter<=50):
             h_theta = sigmoid_x(X_train_class,theta)
             h_theta = h_theta.reshape(y_train_class.shape[0],1)
             theta = theta - (learning_rate*(np.sum(((h_theta-y_train_class)*X_train_class),axis=0)))

             iter+=1
         class_theta.update({int(np.unique(y_train)[0]):np.matrix(theta)})
         #print class_theta
         #exit(0)
         return class_theta

def predicted_y(membership_fn,y_2_k):

    #discri_function = discriminant_function(X,y,fold)


    if(len(np.unique(y_2_k))==2):
        y_predicted = np.zeros(len(membership_fn[0]))
        for i in range(len(y_predicted)):
            if(membership_fn[0][i]>0.5):
                y_predicted[i] = np.unique(y)[1]
            else:
                y_predicted[i] = np.unique(y)[0]
        #print y_predicted
        #exit(0)
        return y_predicted

    else:

        #membership_fn = classify_X(X,y,fold)
        #print membership_fn

        member_list=[]
        for d in membership_fn:
            member_list.append(membership_fn[d])
        max_gjx = np.maximum.reduce(member_list)

        #print max_gjx.shape[0]
        y_predicted = np.zeros(max_gjx.shape[0])
        for y_in in range(max_gjx.shape[0]):
            for key in membership_fn:
                 value_dic = membership_fn[key]
                 if(max_gjx[y_in]==value_dic[y_in]):
                     y_predicted[y_in] = key
        #print y_predicted
        #exit(0)
        return y_predicted

def make_canfusion_matrix(y_test,y_pridct,y):

    mat_size = len(np.unique(y))
    conf_array = np.zeros((mat_size,mat_size),dtype=np.int)
    confusion_matrix =np.matrix(conf_array)
    """if(len(np.unique(y))==3):
        for i in range(len(y_pridct)):
             if(y_pridct[i]==np.unique(y)[0] and y_test[i]==np.unique(y)[0]):
                confusion_matrix[0,0]+=1
             elif(y_pridct[i]==np.unique(y)[1] and y_test[i]==np.unique(y)[0]):
                 confusion_matrix[0,1]+=1
             if(y_pridct[i]==np.unique(y)[1] and y_test[i]==np.unique(y)[1]):
                confusion_matrix[1,1]+=1
             elif(y_pridct[i]==np.unique(y)[0] and y_test[i]==np.unique(y)[1]):
                 confusion_matrix[1,0]+=1"""
    unique_y = np.unique(y)
    for i in range(len(y_pridct)):
        for k in range(len(unique_y)):
            for j in range(len(unique_y)):
                if(y_pridct[i]==unique_y[k] and y_test[i]==unique_y[j]):
                    confusion_matrix[j,k]+=1

    #print confusion_matrix
    #exit(0)
    return confusion_matrix

def parameter_calculation(confusion_mat,y,y_test,y_pridct):

    uniq_y = np.unique(y)#np.unique(train_file[:,-1])
    #MSE = round(np.average((y_pridct-y_test)**2),4)
    Accuracy = round(float(np.matrix.trace(confusion_mat))/np.sum(confusion_mat),4)
    MSE = round((1-Accuracy),4)
    Precision = {}#np.zeros(confusion_mat.shape[0])
    Recall = {}#np.zeros(confusion_mat.shape[0])
    F_measure = {} #np.zeros(confusion_mat.shape[0])


    for i in range(confusion_mat.shape[0]):

        """Precision[i] = round(float(confusion_mat[i,i])/np.sum(confusion_mat[:,i]),4)
        Recall[i] = round(float(confusion_mat[i,i])/np.sum(confusion_mat[i,:]),4)
        F_measure[i] = round(2 * (Precision[i] * Recall[i])/(Precision[i] + Recall[i]),4)"""

        if(np.sum(confusion_mat[:,i])!=0):
            Precision.update({uniq_y[i]:round(float(confusion_mat[i,i])/np.sum(confusion_mat[:,i]),4)})
        else:
            Precision.update({uniq_y[i]:0.0})
        if(np.sum(confusion_mat[i,:])!=0):
            Recall.update({uniq_y[i]:round(float(confusion_mat[i,i])/np.sum(confusion_mat[i,:]),4)})
        else:
            Recall.update({uniq_y[i]:0.0})
        if((Precision[uniq_y[i]] + Recall[uniq_y[i]])!=0):
            F_measure.update({uniq_y[i]:round(2 * (Precision[uniq_y[i]] * Recall[uniq_y[i]])/(Precision[uniq_y[i]] + Recall[uniq_y[i]]),4)})
        else:
            F_measure.update({uniq_y[i]:0.0})


    all_parameter = {}
    all_parameter.update({'mse':MSE})
    all_parameter.update({'accuracy':Accuracy})
    all_parameter.update({'precision':Precision})
    all_parameter.update({'recall':Recall})
    all_parameter.update({'f-measure':F_measure})

    return all_parameter


def print_parameter(all_parameter):


    print "MSE :: ",all_parameter['mse'],"\t\tOR\t\t",100*all_parameter['mse']," %"
    print
    print "Accuracy :: ",all_parameter['accuracy'],"\tOR\t\t",100*all_parameter['accuracy']," %"
    print
    print "Precision :: ",all_parameter['precision']
    print
    print "Recall :: ",all_parameter['recall']
    print
    print "F-measure :: ",all_parameter['f-measure']





def classify_X(X,y,fold):
    z = np.ones((X.shape[0],1), dtype=np.float)
    X = np.c_[z,X]
    kf = KFold(X.shape[0],n_folds=fold,shuffle=True)
    current_fold = 1
    all_fold_parameter = {}
    all_para_inbuilt = {}
    for train_index, test_index in kf:

       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]

       #print np.unique(y_train)
       #exit(0)

       class_theta_train = graident_theta(X_train,y_train)
       #exit(0)
       #print class_theta_train[0].shape
       #print X_test.shape

       max_h_theta = {}
       if(len(np.unique(y))>2):
           for cl in np.unique(y):

               h_theta_value = class_theta_train[cl]*np.matrix(X_test).T
               max_h_theta.update({int(cl):np.asarray(h_theta_value)[0]})

       else:
           h_theta_value = class_theta_train[int(np.unique(y)[0])]*np.matrix(X_test).T
           max_h_theta.update({int(np.unique(y)[0]):np.asarray(h_theta_value)[0]})
       #print max_h_theta
       #exit(0)
       y_pridct = predicted_y(max_h_theta,y)
       conf_matrix =make_canfusion_matrix(y_test,y_pridct,y)
       #print (y_pridct==y_test).sum()
       c_m = confusion_matrix(np.array(y_test),np.array(y_pridct))  # in-built function of confusion matrix.

       all_para = parameter_calculation(conf_matrix,y,y_test,y_pridct)

       lda = LogisticRegression()
       lda.fit(X_train, y_train)
       y_pridct_inb = lda.predict(X_test)

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
    print_parameter(all_fold_parameter[max_accuracy_fold])
    print

    if(len(np.unique(y))==2):
        print
        print "-"*75
        print " "*15," In Built - Parameter Evaluation"
        print "-"*75
        print
        print_parameter(all_para_inbuilt[max_accuracy_fold_inbuilt])







if __name__ == '__main__':

    mnist = fetch_mldata('MNIST original')
    mnist.data.shape
    mnist.target.shape
    np.unique(mnist.target)

    X, y = np.array(mnist.data / 255.), np.array(mnist.target)
    #X_train, X_test = X[:60000], X[60000:]
    #y_train, y_test = y[60000], y[60000:]


    print
    print
    print "-" * 100
    print " " * 15,
    print " 2-class Logistic Regression "
    print "-" * 100


    X_2=np.concatenate((X[y==0],X[y==1]),axis=0)
    y_2=np.concatenate((y[y==0],y[y==1]),axis=0)
    X_2 = np.array(X_2,dtype=np.float)
    y_2 = np.array(y_2,dtype=np.int)



    classify_X(X_2,y_2,10)
    #exit(0)
    print
    print
    print "-" * 100
    print " " * 15,
    print " k-class Logistic Regression  "
    print "-" * 100


    X_mid=np.concatenate((X[y==0],X[y==1]),axis=0)
    X_k=np.concatenate((X_mid,X[y==2]),axis=0)
    y_mid=np.concatenate((y[y==0],y[y==1]),axis=0)
    y_k=np.concatenate((y_mid,y[y==2]),axis=0)
    X_k = np.array(X_k,dtype=np.float)
    y_k = np.array(y_k,dtype=np.int)



    classify_X(X_k,y_k,10)

    """size=len(y_train)

    ## extract "3" digits and show their average"
    ind = [ k for k in range(size) if y_train[k]==9 ]
    extracted_images=X_train[ind,:]

    mean_image=extracted_images.mean(axis=0)
    imshow(mean_image.reshape(28,28), cmap=cm.gray)
    show()"""
