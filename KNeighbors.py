from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris = load_iris()
X, y = iris.data, iris.target

#Display iris data and target
print("iris_data:", iris.data)
print("iris_target:", iris.target)

#spliting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#Display x and y train and test
print("X_trian:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)

#running model and display accuracy

scorces = []
k_values = np.range[1, 10]
for k in k_values:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_test, y_test)
    scorces.append(classifier.scorce(X_test, y_test))


#Dsiplay graph
#plt.plot(k_values, scorces)
#plt.xlabel('Nb neighbors')
#plt.ylabel('accuracy')
#plt.show()



