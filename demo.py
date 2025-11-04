from sklearn import tree
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
clf1 = neighbors.KNeighborsClassifier()
# 2
clf2 = naive_bayes.GaussianNB()
# 3
clf3 = ensemble.RandomForestClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

Y_true = ['male', 'male', 'male', 'female', 'female', 'female']

prediction = clf.predict([[190, 70, 43], [181, 80, 44], [177, 70, 43], [177, 70, 40], [160, 60, 38], [154, 54, 37]])
prediction1 = clf1.predict([[190, 70, 43], [181, 80, 44], [177, 70, 43], [177, 70, 40], [160, 60, 38], [154, 54, 37]])
prediction2 = clf2.predict([[190, 70, 43], [181, 80, 44], [177, 70, 43], [177, 70, 40], [160, 60, 38], [154, 54, 37]])
prediction3 = clf3.predict([[190, 70, 43], [181, 80, 44], [177, 70, 43], [177, 70, 40], [160, 60, 38], [154, 54, 37]])

# CHALLENGE compare their reusults and print the best one!
print("Decision Tree")
print(prediction)
print(accuracy_score(Y_true, prediction))
print("K Nearest Neighbor")
print(prediction1)
print(accuracy_score(Y_true, prediction1))
print("Gaussian NB")
print(prediction2)
print(accuracy_score(Y_true, prediction2))
print("Random Forest")
print(prediction3)
print(accuracy_score(Y_true, prediction3))