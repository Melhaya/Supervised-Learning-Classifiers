import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from visualize_data import prettyPicture
from sklearn.ensemble import RandomForestClassifier
from predict_accuracy import RandomForestAccuracy


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

RandomForestModel = RandomForestClassifier(min_samples_split=10)
RandomForestModel.fit(features_train, labels_train)


accuracy = RandomForestAccuracy(RandomForestModel, features_test, labels_test)
print("Accuracy of AdaBoost classifier is ", accuracy)

prettyPicture(RandomForestModel, features_test, labels_test)

