from prep_terrain_data import makeTerrainData
from visualize_data import prettyPicture
from predict_accuracy import NBAccuracy
from sklearn.naive_bayes import GaussianNB

import numpy as np
import pylab as pl


'''
the training data (features_train, labels_train) have both "fast" and "slow" 
points mixed in together--separate them so we can give them different colors in the 
scatterplot, and visually identify them
'''

features_train, labels_train, features_test, labels_test = makeTerrainData()

grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


NBModel = GaussianNB()
NBModel.fit(features_train, labels_train)


accuracy = NBAccuracy(NBModel, features_test, labels_test)
print("Accuracy of NB classifier is ", accuracy)


prettyPicture(NBModel, features_test, labels_test)




