# Improvements
# - finding optimal k limits

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

SAMPLES = 100000
FEATURES = 2
NO_CLASSES = 3
TEST_SIZE = 0.3
STD_DEV = 1.3


def create_model(model_type, X_train, y_train, X_test, y_test, k):
    model_name = type(model_type).__name__
    time_start = time.perf_counter()
    model = model_type.fit(X_train, y_train)
    time_elapsed = (time.perf_counter() - time_start)

    prob = model_type.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    print(f"{model_name} Metrics")
    print("Computation Time:%5.4f seconds" % time_elapsed)
    print("Accuracy: %.2f" % accuracy, "\n")
    if model_name == 'KNeighborsClassifier':
        print("Optimal K value: %f" % k)
    confusion_matrix(model, model_name, X_test, y_test)


def confusion_matrix(model, model_name, X_test, y_test):
    plot_confusion_matrix(model, X_test, y_test, normalize='true')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()


# Generating 2D 3-class classification dataset using sklearn function
X, y = make_blobs(n_samples=SAMPLES, n_features=FEATURES, centers=None, cluster_std=STD_DEV,
                  shuffle=True, random_state=68, return_centers=False)
# Plot generated dataset
colours = ['red', 'green', 'purple']
plt.scatter(X[:, 0], X[:, 1], s=0.5, c=y, cmap=ListedColormap(colours))
plt.title('Generated 2D 3-class Dataset')
plt.show()

# Subset data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=2)

# # k-Nearest Neighbour
k_acc_scores = []
k_values = [i for i in range(1, int(SAMPLES / 50000), 2)]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    cross_scores = cross_val_score(knn, X_train, y_train, cv=None, scoring='accuracy')
    k_acc_scores.append(cross_scores.mean())
optimal_k = k_values[k_acc_scores.index(max(k_acc_scores))]

models = [GaussianNB(), LogisticRegression(), KNeighborsClassifier(n_neighbors=optimal_k)]
for model in models:
    create_model(model, X_train, y_train, X_test, y_test, k)
