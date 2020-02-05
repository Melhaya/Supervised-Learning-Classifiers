import sys
from visualize_data import prettyPicture
from prep_terrain_data import makeTerrainData
from predict_accuracy import *
import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl
from sklearn.svm import SVC


features_train, labels_train, features_test, labels_test = makeTerrainData()

#SVM doesnt work very well with large dataset and noisy dataset

########################## SVM #################################

SVMmodel = SVC(kernel="linear", gamma=0.1, C=1)   #Bigger C cause the network to overfit the training set
						  # Gamma defines how far the influence of a single training example reaches (Low value -> far reach)

SVMmodel.fit(features_train, labels_train)
accuracy = SVMccuracy(SVMmodel, features_test, labels_test)

print("Accuracy of SVM classifier is ", accuracy)

prettyPicture(SVMmodel, features_test, labels_test)


