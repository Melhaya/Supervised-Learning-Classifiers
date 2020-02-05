import sys
from visualize_data import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from sklearn import tree
from predict_accuracy import DTAccuracy

# Decision trees solve datasets that are not seperable linearly with linear lines.
# Entropy: measure of impurity in a bunch of examples. 
# Entropy = sum(-p_i *log2(p_i))  0 means its pure   1 means impure
# Information gain = entropy(parent) - (weighted average)*Entropy(Children).... DT maximizes Information gain
# Strength: easy to use, graphically interpret data better.
# Weakness: Prone to overfitting (Specially with many features)


features_train, labels_train, features_test, labels_test = makeTerrainData()


DTModel = tree.DecisionTreeClassifier(min_samples_split=50)
DTModel.fit(features_train, labels_train)

accuracy = DTAccuracy(DTModel, features_test, labels_test)
print("Accuracy of Decision Tree classifier is ", accuracy)

prettyPicture(DTModel, features_test, labels_test)
