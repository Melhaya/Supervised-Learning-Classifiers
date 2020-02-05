import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def NBAccuracy(clf, features_test, labels_test):
    """ compute the accuracy of the Naive Bayes classifier """

    pred = clf.predict(features_test)
    return clf.score(features_test,labels_test)
#	from sklearn.metrics import accuracy_score
#	accuracy_score(pred,labels_test)

def SVMccuracy(clf, features_test, labels_test):

	pred = clf.predict(features_test)
	acc = accuracy_score(pred, labels_test)
	return acc

def DTAccuracy(clf, features_test, labels_test):

	pred = clf.predict(features_test)
	acc = accuracy_score(pred, labels_test)
	return acc

def KNNAccuracy(clf, features_test, labels_test):

	pred = clf.predict(features_test)
	acc = accuracy_score(pred, labels_test)
	return acc	

def AdaBoostAccuracy(clf, features_test, labels_test):

	pred = clf.predict(features_test)
	acc = accuracy_score(pred, labels_test)
	return acc

def RandomForestAccuracy(clf, features_test, labels_test):

	pred = clf.predict(features_test)
	acc = accuracy_score(pred, labels_test)
	return acc