import numpy as np
from sklearn.cross_validation import KFold
from sklearn.datasets import load_iris
import math
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,f1_score,recall_score,precision_recall_curve,average_precision_score
import Logistic_Regression as LR
#from sklearn.neural_network import MLPClassifier


#iris = load_iris()


def sigmoid_x(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_output_to_derivative(out):
    return out * (1 - out)


def classify_X(X,y,weight_xy,fold):
    kf = KFold(X.shape[0],n_folds=fold,shuffle=True)
    current_fold = 1
    all_fold_parameter = {}
    all_para_inbuilt = {}
    for train_index, test_index in kf:

       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]

       #print X_test.shape
       #print weight_xy.shape
       #exit(0)

       for iter in xrange(10000):

        layer_0 = X_test
        layer_1 = sigmoid_x(np.dot(layer_0, weight))

        layer_1_error = layer_1 - y_test

        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        synapse_0_derivative = np.dot(layer_0.T, layer_1_delta)

        weight_xy -= synapse_0_derivative

       y_pridct = map(lambda x:math.ceil(x),layer_1)
       #y_pridct = predicted_y(max_h_theta,y)
       conf_matrix =LR.make_canfusion_matrix(y_test,y_pridct,y)
       #print (y_pridct==y_test).sum()
       c_m = confusion_matrix(np.array(y_test),np.array(y_pridct))  # in-built function of confusion matrix.

       all_para = LR.parameter_calculation(conf_matrix,y,y_test,y_pridct)

       #lda = LogisticRegression()
       #lda.fit(X_train, y_train)
       #y_pridct_inb = lda.predict(X_test)


       mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    algorithm='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

       mlp.fit(X_train, y_train)
       print("Training set score: %f" % mlp.score(X_train, y_train))
       print("Test set score: %f" % mlp.score(X_test, y_test))
       y_pridct_inb = mlp.predict(X_test)


       all_parameter = {}
       all_parameter.update({'mse':round((np.average((y_pridct_inb-y_test)**2)),4)})
       all_parameter.update({'accuracy':round(accuracy_score(y_test,y_pridct_inb),4)})
       all_parameter.update({'precision':round(precision_score(y_test,y_pridct_inb),4)})
       all_parameter.update({'recall':round(recall_score(y_test,y_pridct_inb),4)})
       all_parameter.update({'f-measure':round(f1_score(y_test,y_pridct_inb),4)})
       


       all_fold_parameter.update({current_fold:all_para})
       #all_para_inbuilt.update({current_fold:all_parameter})

       current_fold+=1

    max_accuracy_fold = max(all_fold_parameter,key = lambda k:float(all_fold_parameter[k]['accuracy']))

    #max_accuracy_fold_inbuilt = max(all_para_inbuilt,key = lambda k:float(all_para_inbuilt[k]['accuracy']))
    print
    print "-"*75
    print " "*15,"Parameter Evaluation"
    print "-"*75
    print
    LR.print_parameter(all_fold_parameter[max_accuracy_fold])
    print

    if(len(np.unique(y))==2):
        print
        print "-"*75
        print " "*15," In Built - Parameter Evaluation"
        print "-"*75
        print
        #LR.print_parameter(all_para_inbuilt[max_accuracy_fold_inbuilt])




if __name__ == '__main__':

    iris = load_iris()
    X, y = iris.data, iris.target

    y = y.reshape(y.shape[0], 1)

    print "X shape: ",X.shape
    print "Y shape: ",y.shape

    weight = np.zeros((X.shape[1], 1))

    print
    print
    print "-" * 100
    print " " * 15,
    print " k-class Multi-Layer Perception "
    print "-" * 100


    classify_X(X,y,weight,20)
