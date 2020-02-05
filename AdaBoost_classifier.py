import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from visualize_data import prettyPicture
from sklearn.ensemble import AdaBoostClassifier
from predict_accuracy import AdaBoostAccuracy


features_train, labels_train, features_test, labels_test = makeTerrainData()


grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
#plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 1.0)
#plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
#plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
#plt.legend()
#plt.xlabel("bumpiness")
#plt.ylabel("grade")
#plt.show()
################################################################################

AdaBoostModel = AdaBoostClassifier( learning_rate=0.4)
AdaBoostModel.fit(features_train, labels_train)


accuracy = AdaBoostAccuracy(AdaBoostModel, features_test, labels_test)
print("Accuracy of AdaBoost classifier is ", accuracy)

prettyPicture(AdaBoostModel, features_test, labels_test)

