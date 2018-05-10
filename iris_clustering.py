from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

iris = load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_trian, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)

k=0
for k in range(1, 10):
    kmeans = KMeans(n_cluster=k, random_state=42)

    labels = kmeans.fit_predict(X)

    print("clustering iris data:\n", labels)
