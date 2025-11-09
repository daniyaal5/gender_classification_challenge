from sklearn import tree
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn.metrics import accuracy_score
import numpy as np

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1 K Nearest Neighbor
clf_KNN = neighbors.KNeighborsClassifier()
# 2 Naive Bayes Gaussian
clf_NB = naive_bayes.GaussianNB()
# 3 Random Forest
clf_RF = ensemble.RandomForestClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# CHALLENGE - ...and train them on our data

# Classifiers
clf = clf.fit(X, Y)
clf_KNN = clf_KNN.fit(X, Y)
clf_NB = clf_NB.fit(X, Y)
clf_RF = clf_RF.fit(X, Y)

# Test dataset
test_data = [[190, 70, 43], [181, 80, 44], [177, 70, 43], [177, 70, 40], [160, 60, 38], [154, 54, 37]]

# True values for the test dataset
Y_true = ['male', 'male', 'male', 'female', 'female', 'female']

# Predictions
prediction = clf.predict(test_data)
pred_KNN = clf_KNN.predict(test_data)
pred_NB = clf_NB.predict(test_data)
pred_RF = clf_RF.predict(test_data)

# Accuracy Scores
acc = accuracy_score(Y_true, prediction)
acc_KNN = accuracy_score(Y_true, pred_KNN)
acc_NB = accuracy_score(Y_true, pred_NB)
acc_RF = accuracy_score(Y_true, pred_RF)

# Storing results in a dictionary
results = {
    "K Nearest Neighbor" : [pred_KNN, acc_KNN],
    "Naive Bayes Gaussian" : [pred_NB, acc_NB],
    "Random Forest" : [pred_RF, acc_RF]
}

accuracies = []

for key, value in results.items():
    print(key)
    print(value)

best_classifier = max(results.items(), key= lambda item: item[1][1])
print(f"Best classifier is {best_classifier[0]}")