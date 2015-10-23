import numpy as np
import matplotlib.pyplot as plt
import loadFile   # Custom library

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import  make_scorer     # To make a scorer for the GridSearch.
from collections import deque
from sklearn.metrics import accuracy_score

plt.close('all')

Data_Prep = 1
SVM_cl = 0
KNN_cl = 0
GNB_cl = 1
Tree_cl = 0
LDA_cl = 0

#%%load kinect normalized data
data, labels = loadFile.load_data("skeletonData.txt")
loadFile.show_info(labels) #print data info

#################################################################
#################### DATA PREPROCESSING #########################
#################################################################
if (Data_Prep == 1):

    from sklearn import preprocessing
    from sklearn import cross_validation
    #randomize the samples index and divide between Training and Testing data
    Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(data, labels, test_size = 0.2, random_state=0)

    #Standarize the features with train data: zero mean and unit variance
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

#################################################################
###################### CLASSIFIERS ##############################
#################################################################
if (Tree_cl == 1):
    param_grid = dict()
    param_grid.update({'max_features':[None,'auto']})
    param_grid.update({'max_depth':np.arange(1,21)})
    param_grid.update({'min_samples_split':np.arange(2,11)})
    gtree = GridSearchCV(DecisionTreeClassifier(),param_grid,scoring='precision',cv=StratifiedKFold(Ytrain, n_folds = 5),refit=True,n_jobs=-1)
    gtree.fit(Xtrain,Ytrain)
    scores = np.empty((6))
    scores[0] = gtree.score(Xtrain,Ytrain)
    scores[1] = gtree.score(Xtest,Ytest)
    print "---------------------Decission Tree Clasifier--------------"
    print('Decision Tree, train: {0:.02f}% '.format(scores[0]*100))
    print('Decision Tree, test: {0:.02f}% '.format(scores[1]*100))

if (LDA_cl == 1):
    from sklearn.lda import LDA
    lda = LDA()
    lda.fit(Xtrain,Ytrain)
    scores = np.empty((4))
    scores[0] = lda.score(Xtrain,Ytrain)
    scores[1] = lda.score(Xtest,Ytest)
    print "---------------------Linear Discriminant Analysis---------------------------"
    print('LDA, train: {0:.02f}% '.format(scores[0]*100))
    print('LDA, test: {0:.02f}% '.format(scores[1]*100))

    print lda.predict_proba(Xtest[-1])
    print lda.predict(Xtest[-1])
    print Ytest[-1]

if (GNB_cl == 1):
    nb = GaussianNB()
    nb.fit(Xtrain,Ytrain)
    scores = np.empty((4))
    scores[0] = nb.score(Xtrain,Ytrain)
    scores[1] = nb.score(Xtest,Ytest)

    print "---------------------Naive Bayes Classifier------------------"
    print "Prediction time:", t1-t0, "s"
    print('Gaussian Naive Bayes, train: {0:.02f}% '.format(scores[0]*100))
    print('Gaussian Naive Bayes, test: {0:.02f}% '.format(scores[1]*100))

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

    trainscores = [gsvml.score(Xtrain,Ytrain),gsvmp.score(Xtrain,Ytrain),gsvmr.score(Xtrain,Ytrain)]
    testscores = [0,0,0,0]
    testscores[0] = gsvml.score(Xtest,Ytest)
    testscores[1] = gsvmp.score(Xtest,Ytest)
    testscores[2] = gsvmr.score(Xtest,Ytest)
    testscores[3] = gsvmr.score(Xtrain,Ytrain)
    maxtrain = np.amax(trainscores)
    maxtest = np.amax(testscores)
    print "---------------------Support Vector Classifier---------------"
    print('Linear SVM(C = {0:.02f}), score: {1:.02f}% '.format(gsvml.best_params_['C'], testscores[0]*100))
    print('Poly SVM(C = {0:.02f}, degree = {1:.02f}), score: {2:.02f}% '.format(gsvmp.best_params_['C'], gsvmp.best_params_['degree'], testscores[1]*100))
    print('rbf SVM(C = {0:.02f}, gamma = {1:.02f}), score: {2:.02f}% '.format(gsvmr.best_params_['C'],gsvmr.best_params_['gamma'] , testscores[2]*100) )

    #################################################
    ####    PLOTTING GRID Search SCORES       #######
    #################################################


    #GRID SEARCH plotting for ploy kernel

    grid_scores_gsvmp = {} #scores and C values for each degree value tested
    for i in range(len(gsvmp.grid_scores_)):
        degree_aux = gsvmp.grid_scores_[i][0]['degree']
        c_aux = gsvmp.grid_scores_[i][0]['C']
        score = gsvmp.grid_scores_[i][1]
        if degree_aux not in grid_scores_gsvmp:
            grid_scores_gsvmp[degree_aux] = {'scores':[], 'C':[]}
        grid_scores_gsvmp[degree_aux]['scores'].append(score)
        grid_scores_gsvmp[degree_aux]['C'].append(c_aux)

    plt.figure()
    plt.plot(grid_scores_gsvmp[2]['C'],grid_scores_gsvmp[2]['scores'],c='g',lw=2,aa=True, label='degree = 2')
    plt.plot(grid_scores_gsvmp[3]['C'],grid_scores_gsvmp[3]['scores'],c='b',lw=2,aa=True, label='degree = 3')
    plt.plot(grid_scores_gsvmp[4]['C'],grid_scores_gsvmp[4]['scores'],c='r',lw=2,aa=True, label='degree = 4')
    plt.title('Grid precission for C in SVM_poly')
    plt.xlabel('C')
    plt.ylabel('Mean training precission')
    plt.xscale('log')
    plt.legend(loc=4)
    plt.grid()
    plt.show()

    #GRID SEARCH plotting for rbf kernel

    grid_scores_gsvmr = {}
    for i in range(len(gsvmr.grid_scores_)):
        score = gsvmr.grid_scores_[i][1]
        gammas = float("{0:.6f}".format(gsvmr.grid_scores_[i][0]['gamma']))
        c_aux = gsvmr.grid_scores_[i][0]['C']
        if gammas not in grid_scores_gsvmr:
            grid_scores_gsvmr[gammas] = {'scores':[], 'C':[]}
        grid_scores_gsvmr[gammas]['scores'].append(score)
        grid_scores_gsvmr[gammas]['C'].append(c_aux)

    plt.figure()
    plt.plot(grid_scores_gsvmr[0.01]['C'],grid_scores_gsvmr[0.01]['scores'],c='g',lw=2,aa=True, label='gamma = 0.01')
    plt.plot(grid_scores_gsvmr[0.005]['C'],grid_scores_gsvmr[0.005]['scores'],c='r',lw=2,aa=True, label='gamma = 0.005')
    plt.plot(grid_scores_gsvmr[0.000625]['C'],grid_scores_gsvmr[0.000625]['scores'],c='b',lw=2,aa=True, label='gamma = 0.000625')
    plt.plot(grid_scores_gsvmr[0.02]['C'],grid_scores_gsvmr[0.02]['scores'],c='y',lw=2,aa=True, label='gamma = 0.02')
    plt.title('Grid precission for C in SVM_rbf')
    plt.xlabel('C')
    plt.ylabel('Mean training precission')
    plt.xscale('log')
    plt.legend(loc=4)
    plt.grid()
    plt.show()

if (KNN_cl == 1):

    # Perform authomatic grid search
    params = [{'n_neighbors':np.arange(1,11)}]
    gknn = GridSearchCV(KNeighborsClassifier(),params,scoring='f1_weighted',cv=StratifiedKFold(Ytrain, n_folds = 5),refit=True,n_jobs=-1)

    gknn.fit(Xtrain,Ytrain)
    scores = np.empty((4))
    scores[0] = gknn.score(Xtrain,Ytrain)
    scores[1] = gknn.score(Xtest,Ytest)

    print "---------------------K-NN Classifier---------------------------"
    print "Prediction time:", t1-t0,"s"
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

    print cross_validate(gknn,Xtest,Ytest)
