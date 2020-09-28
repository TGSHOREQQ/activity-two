from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,plot_confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

SAMPLES = 100000
FEATURES = 2
NO_CLASSES = 3
TEST_SIZE = 0.3
STD_DEV = 1.3

# Generating 2D 3-class classification dataset using sklearn function
X, y = make_blobs(n_samples=SAMPLES, n_features=FEATURES, centers=None, cluster_std=STD_DEV,
                  shuffle=True, random_state=68, return_centers=False)
colours = ['red', 'green', 'purple']

plt.scatter(X[:, 0], X[:, 1], s=0.5, c=y, cmap=ListedColormap(colours))
plt.title('Generated 2D 3-class Dataset')
plt.show()

# Subset data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=2)

# Gaussian Naive Bayes
gnb = GaussianNB()
time_start_gnb = time.perf_counter()
gnb_model = gnb.fit(X_train, y_train)
time_elapsed_gnb = (time.perf_counter() - time_start_gnb)
y_pred_gnb = gnb_model.predict(X_test)
gnb_accuracy = accuracy_score(y_test, y_pred_gnb) * 100
plot_confusion_matrix(gnb_model, X_test, y_test, normalize='true')
plt.title('Gaussian Naive Bayes Confusion Matrix')
plt.show()

# Logistic Regression
lr = LogisticRegression()
time_start_lr = time.perf_counter()
lr_model = lr.fit(X_train, y_train)
time_elapsed_lr = (time.perf_counter() - time_start_lr)
y_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr) * 100
plot_confusion_matrix(lr_model, X_test, y_test, normalize='true')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# # Multivariate Gaussian





# # k-Nearest Neighbour
k_acc_scores = []
k_values = [i for i in range(1, int(SAMPLES/2000), 2)]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    cross_scores = cross_val_score(knn, X_train, y_train, cv=None, scoring='accuracy')
    k_acc_scores.append(cross_scores.mean())

optimal_k = k_values[k_acc_scores.index(max(k_acc_scores))]
print(optimal_k)
knn = KNeighborsClassifier(n_neighbors=optimal_k, n_jobs=-1)
time_start_knn = time.perf_counter()
knn_model = knn.fit(X_train, y_train)
time_elapsed_knn = (time.perf_counter() - time_start_knn)
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn) * 100
plot_confusion_matrix(knn_model, X_test, y_test, normalize='true')
plt.title('kNN Confusion Matrix')
plt.show()


# function for repeatable?
# Gaussian Naive Bayes Metrics
print("Gaussian Naive Bayes Metrics")
print("GNB Computation Time:%5.4f seconds" % time_elapsed_gnb)
print("GNB Accuracy: %.4f" % gnb_accuracy, "\n")
confusion_matrix(y_test, y_pred_gnb)

# Logistic Regression Metrics
print("Logistic Regression Metrics")
print("LR Computation Time:%5.4f seconds" % time_elapsed_lr)
print("LR Accuracy: %.4f" % lr_accuracy, "\n")

# kNN Metrics
print("kNN Metrics")
print("kNN Computation Time:%5.4f seconds" % time_elapsed_knn)
print("kNN Accuracy: %.4f" % knn_accuracy, "\n")

# # Multivariate Gaussian Metrics
# print("Multivariate Gaussian Metrics")
# print("MG Computation Time:%5.4f seconds" % time_elapsed_mg)
# print("MG Accuracy: %.4f" % mg_accuracy)
