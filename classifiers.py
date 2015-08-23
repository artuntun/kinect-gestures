import numpy as np
import matplotlib.pyplot as plt
import loadFile

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import  make_scorer     # To make a scorer for the GridSearch.
from collections import deque
from sklearn.metrics import accuracy_score

plt.close('all')

Data_Prep = 0;
SVM_cl = 0;
KNN_cl = 1;

#%%load kinect normalized data
data, labels = loadFile.load_data("skeletonData.txt")
loadFile.show_info(labels) #print data info

#################################################################
#################### DATA PREPROCESSING #########################
#################################################################
if (Data_Prep == 1):

    #%% Split data in training and test sets
    train_ratio = 0.8
    rang = np.arange(np.shape(data)[0],dtype=int) # Create array of index
    np.random.seed(0)
    rang = np.random.permutation(rang)        # Randomize the array of index

    Ntrain = round(train_ratio*np.shape(data)[0])    # Number of samples used for training
    Ntest = len(rang)-Ntrain                  # Number of samples used for testing
    Xtrain = data[rang[:Ntrain]]
    Xtest = data[rang[Ntrain:]]
    Ytrain = labels[rang[:Ntrain]]
    Ytest = labels[rang[Ntrain:]]

    #%% Normalize data
    mx = np.mean(Xtrain,axis=0,dtype=np.float64)
    stdx = np.std(Xtrain,axis=0,dtype=np.float64)

    Xtrain = np.divide(Xtrain-np.tile(mx,[len(Xtrain),1]),np.tile(stdx,[len(Xtrain),1]))
    Xtest = np.divide(Xtest-np.tile(mx,[len(Xtest),1]),np.tile(stdx,[len(Xtest),1]))

    nClasses = np.alen(np.unique(labels))
    nFeatures = np.shape(data)[1]

else:
    #==============================================================================
    # # Also we could have used:
    from sklearn import preprocessing
    from sklearn import cross_validation
    Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(data, labels, test_size = 0.2, random_state=0)
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    #==============================================================================

#################################################################
###################### CLASSIFIERS ##############################
#################################################################

if (SVM_cl == 1):

    # Parameters for the validation
    C = np.logspace(-3,3,10)
    p = np.arange(2,5)
    gamma = np.array([0.125,0.25,0.5,1,2,4])/200

    # Create dictionaries with the Variables for the validation !
    # We create the dictinary for every TYPE of SVM we are gonna use.
    param_grid_linear = dict()
    param_grid_linear.update({'kernel':['linear']})
    param_grid_linear.update({'C':C})

    param_grid_pol = dict()
    param_grid_pol.update({'kernel':['poly']})
    param_grid_pol.update({'C':C})
    param_grid_pol.update({'degree':p})

    param_grid_rbf = dict()
    param_grid_rbf.update({'kernel':['rbf']})
    param_grid_rbf.update({'C':C})
    param_grid_rbf.update({'gamma':gamma})


    param = [{'kernel':'linear','C':C}]
    param_grid = [param_grid_linear,param_grid_pol,param_grid_rbf]

    # Validation is useful for validating a parameter, it uses a subset of the
    # training set as "test" in order to know how good the generalization is.
    # The folds of "StratifiedKFold" are made by preserving the percentage of samples for each class.
    stkfold = StratifiedKFold(Ytrain, n_folds = 5)

    # The score function is the one we want to minimize or maximize given the label and the predicted.
    acc_scorer = make_scorer(accuracy_score)

    # GridSearchCV implements a CV over a variety of Parameter values !!
    # In this case, over C fo the linear case, C and "degree" for the poly case
    # and C and "gamma" for the rbf case.
    # The parameters we have to give it are:
    # 1-> Classifier Object: SVM, LR, RBF... or any other one with methods .fit and .predict
    # 2 -> Subset of parameters to validate. C
    # 3 -> Type of validation: K-fold
    # 4 -> Scoring function. sklearn.metrics.accuracy_score

    gsvml = GridSearchCV(SVC(class_weight='auto'),param_grid_linear, scoring = acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
    gsvmp = GridSearchCV(SVC(class_weight='auto'),param_grid_pol, scoring = acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
    gsvmr = GridSearchCV(SVC(class_weight='auto'),param_grid_rbf, scoring =acc_scorer,cv = stkfold, refit = True,n_jobs=-1)

    gsvml.fit(Xtrain,Ytrain)
    gsvmp.fit(Xtrain,Ytrain)
    gsvmr.fit(Xtrain,Ytrain)

    params = gsvmr.get_params() #parameters found with gridSerach of the rbf classifier

    trainscores = [gsvml.score(Xtrain,Ytrain),gsvmp.score(Xtrain,Ytrain),gsvmr.score(Xtrain,Ytrain)]
    testscores = [gsvml.score(Xtest,Ytest),gsvmp.score(Xtest,Ytest),gsvmr.score(Xtest,Ytest)]
    maxtrain = np.amax(trainscores)
    maxtest = np.amax(testscores)
    print('Linear SVM, score: {0:.02f}% '.format(testscores[0]*100))
    print('Poly SVM, score: {0:.02f}% '.format(testscores[1]*100))
    print('rbf SVM, score: {0:.02f}% '.format(testscores[2]*100))

if (KNN_cl == 1):

    # Perform authomatic grid search
    params = [{'n_neighbors':np.arange(1,11)}]
    gknn = GridSearchCV(KNeighborsClassifier(),params,scoring='f1_weighted',cv=StratifiedKFold(Ytrain, n_folds = 5),refit=True,n_jobs=-1)

    gknn.fit(Xtrain,Ytrain)
    scores = np.empty((4))
    scores[0] = gknn.score(Xtrain,Ytrain)
    scores[1] = gknn.score(Xtest,Ytest)
    print('{0}-NN, train: {1:.02f}% '.format(gknn.best_estimator_.n_neighbors,scores[0]*100))
    print('{0}-NN, test: {1:.02f}% '.format(gknn.best_estimator_.n_neighbors,scores[1]*100))

    ##### Scores in validation, training and testing data for each K-nn tested
    tested_Ks = [val[0]['n_neighbors'] for val in gknn.grid_scores_] # K tested in GridSearch
    scores_validation = [val[1] for val in gknn.grid_scores_]  #Scores at validation
    scores_training = []
    scores_testing = []
    for x in tested_Ks:
        testing_clf = KNeighborsClassifier(n_neighbors = x, weights='uniform')
        testing_clf.fit(Xtrain,Ytrain)
        scores_testing.append(testing_clf.score(Xtest,Ytest))
        scores_training.append(testing_clf.score(Xtrain,Ytrain))

    #Plot evolution of parameters performance
    plt.figure()
    plt.plot(tested_Ks,scores_validation,c='g',lw=2,aa=True, label='Validation')
    plt.plot(tested_Ks,scores_training,c='b',lw=2,aa=True, label='Training')
    plt.plot(tested_Ks,scores_testing,c='r',lw=2,aa=True, label= 'Testing')
    plt.plot(np.argmax(scores)+1,np.amax(scores),'v')
    plt.title('Grid precission for k in kNN')
    plt.xlabel('k')
    plt.ylabel('Mean training precission')
    plt.legend()
    plt.grid()
    plt.show()
