from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

mdl = tree.DecisionTreeClassifier().fit(iris.data, iris.target)

pred = mdl.predict(iris.data)
