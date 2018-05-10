from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

iris = load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0, stratify=iris.target)

#clustering the data
kmeans = KMeans(n_cluster=3, random_state=42)

labels = kmeans.fit_predict(X)

#Display the cluter 
print("---Display The cluster of the iris data---")

print("Cluster of the iris data:\n", labels)
