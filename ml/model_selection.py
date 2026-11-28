import numpy as np
from sklearn import datasets
from sklearn import neighbors
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Here we will see how to select a model in the first place
# Studying bias variance tradeoff we understood how to manage underfittig and overfitting
# and how to evalute a model

# We will see how to select a model in the first place with these strategies:
# 1. Feature selection, which is a supervised methodology to identify the relevant features for a given task;
# 2. Principal component analysis for dimensionality reduction,
# which is an unsupervised methodology to reduce the dataset dimensionality;
# 3. Regularization with the  k  paramater in the  k -nearest neighbours classifier.


# 1. Feature Selection

# We have Iris dataset and we want to train a model to classify 'Iris-virginica' against 'Non-virginica'

iris = datasets.load_iris()

# 150 flowers of 3 iris species, 50 of each:

# 0 = Setosa
# 1 = Versicolor
# 2 = Virginica

# Each flower is described by 4 features, all measured in cm:

# x₀ = sepal length
# x₁ = sepal width
# x₂ = petal length
# x₃ = petal width

# the variable iris now contains inputs ('data' field), targets ('target', field)
# and other stuff. Let us keep the inputs and targets in two separate variables

X = iris.data
y = iris.target
print(X.shape, y.shape)
# (150, 4) (150,)
# X has 150 row and 4 columns, y has 150 value


print('Original targets\n', y)
# There are 0 1 2

# Now I do a copy of the dataset with only 100 flower,
# 50 versicolar and 50 virginica

y = iris.target.copy()
# 0 - setosa, 1 - versicolor, 2 - virginica
y[y == 1] = 0
y[y == 2] = 1

print('Iris-Virginica targets\n', y)
# There are 0 1

# set the seed to get consistent splits
np.random.seed(0)

# let us concatenate X and y in a single matrix
D = np.concatenate((X, y.reshape(len(y), -1)), axis=1)

# then we shuffle D
np.random.shuffle(D)

# and we slip the data as 60/20/20
X_train = D[:90,:4]
y_train = D[:90,4]
X_vali = D[90:120,:4]
y_vali = D[90:120,4]
X_test = D[120:150,:4]
y_test = D[120:150,4]


# logistic regression on the original dataset
log_classifier = LogisticRegression(penalty=None) # regularization is applied by default
log_classifier.fit(X_train, y_train)
y_hat_vali = log_classifier.predict(X_vali)
vali_accuracy = sum(y_hat_vali == y_vali) / len(y_vali)
print('Validation accuracy:', vali_accuracy)

# now I do 4 model, each without a feature, 
# I test the accurancy of each model

# full set of features
F = [0, 1, 2, 3]

for i in range(len(F)):
  # remove the feature i from train and validation
  X_train_i = np.delete(X_train, i, axis=1)
  X_vali_i = np.delete(X_vali, i, axis=1)
  # fit the classifier
  log_classifier_i = LogisticRegression(penalty=None) # regularization is applied as default
  log_classifier_i.fit(X_train_i, y_train)
  # evaluate on validation
  y_hat_vali = log_classifier_i.predict(X_vali_i)
  vali_accuracy = sum(y_hat_vali == y_vali) / len(y_vali)
  print('The model with features', F, 'without', F[i], 'has validation accuracy:', vali_accuracy)


# all the model have validation accurancy 1
# removing one feature does not seem to impact the performance

# we remove a feature adn we repeat the process

F = [1, 2, 3]
X_train_new = X_train[:, F]
X_vali_new = X_vali[:, F]
for i in range(len(F)):
  # remove the feature from train and validation
  X_train_i = np.delete(X_train_new, i, axis=1)
  X_vali_i = np.delete(X_vali_new, i, axis=1)
  # fit the classifier
  log_classifier_i = LogisticRegression(penalty=None) # regularization is applied as default
  log_classifier_i.fit(X_train_i, y_train)
  # evaluate on validation
  y_hat_vali = log_classifier_i.predict(X_vali_i)
  vali_accuracy = sum(y_hat_vali == y_vali) / len(y_vali)
  print('The model with features', F, 'without', F[i], 'has validation accuracy:', vali_accuracy)

# removing feature 1 we still have perfect accurancy, removing feature 2 or 3 we don't
# so we remove feature 1 and we repeat

F = [2, 3]
X_train_new = X_train[:,F]
X_vali_new = X_vali[:, F]
for i in range(len(F)):
  # remove the feature from train and validation
  X_train_i = np.delete(X_train_new, i, axis=1)
  X_vali_i = np.delete(X_vali_new, i, axis=1)
  # fit the classifier
  log_classifier_i = LogisticRegression(penalty=None) # regularization is applied as default
  log_classifier_i.fit(X_train_i, y_train)
  # evaluate on validation
  y_hat_vali = log_classifier_i.predict(X_vali_i)
  vali_accuracy = sum(y_hat_vali == y_vali) / len(y_vali)
  print('The model with features', F, 'without', F[i], 'has validation accuracy:', vali_accuracy)


# we obtain same accurancy, we maintain both and we stop

# now we check the performance on the training set

# we can merge back the validation into the train set
X_train = np.concatenate((X_train, X_vali))
y_train = np.concatenate((y_train, y_vali))

# and extract the dataset after feature selection
F = [2, 3]
X_train_fs = X_train[:, F]
X_test_fs = X_test[:, F]

# we fit the model once again on the new training
log_classifier_fs = LogisticRegression(penalty=None)
log_classifier_fs.fit(X_train_fs, y_train)
# and we evaluate it on the test set
y_hat_test = log_classifier_fs.predict(X_test_fs)
test_accuracy = sum(y_hat_test == y_test) / len(y_test)
print('The model with feature', F, 'has test accuracy:', test_accuracy)


virginica = X[y == 1]
not_virginica = X[y == 0]


def plot_ds(X, w, step=100, label='DS'):
  ds_x1 = np.linspace(X[:,2].min(), X[:,2].max(), step)
  ds_x2 = [-(w[0] + w[1]*x1) / w[2] for x1 in ds_x1]
  plt.plot(ds_x1, ds_x2, label=label)

plt.figure(figsize=(12,7))
coef = log_classifier_fs.coef_.flatten() # weights
w0 = log_classifier_fs.intercept_[0] # bias
log_w = np.array([w0, coef[0], coef[1]])
plot_ds(X, log_w, label='Logistic Regression')

plt.scatter(virginica[:, 2], virginica[:, 3], label='virginica')
plt.scatter(not_virginica[:, 2], not_virginica[:, 3], label='not virginica', marker='x')

plt.xlabel('x2')
plt.ylabel('x3')
plt.grid()
plt.legend()
plt.savefig("ml/plot/ms_fig1.png", dpi=150)


# 2. Dimensionality Reduction

# We use Principal Component Analysis (PCA), an unsupervised technique,
# to extract some low dimensional features from a dataset

iris = datasets.load_iris()

# the variable iris now contains inputs ('data' field), targets ('target', field)
# and other stuff. Let us keep the inputs and targets in two separate variables

X = iris.data
y = iris.target

# Note: PCA is an unsupervised technique, so we do not need y actually

# To identify the principal components (i.e., the extracted features), we have to:

# 1. Center the original inputs X to obtain a matrix ( X_tilde ) with zero (column-wise) mean;
# 2. Compute the covariance matrix of  X , i.e.,  C=X_tilde⊤X_tilde ;
# 3. Compute the eigenvectors  e  and eigenvalues  λ  of  C :

X_tilde = X - np.mean(X, axis=0)

# Note: np.mean(X, axis=0) returns a vector containing the mean of each column

C = np.dot(X_tilde.T, X_tilde)

print(C)

eigenvalues, eigenvectors = np.linalg.eigh(C)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

print('Eigenvalues:\n', eigenvalues)
print('Eigenvectors:\n', eigenvectors)
print('Percentage of explained variance:', eigenvalues / np.sum(eigenvalues))

# Note: each column of eigenvectors is the unit-length eigenvector corresponding to the i-th eigenvalue in eigenvalues

W = eigenvectors

T = np.dot(X_tilde, W)

print('Original data point:', X[21])
print('Transformed data point:', T[21])

# ususally it's not compute by hand but it's simply used the sklearn implementation

# we create a PCA object
pca = PCA()

# we fit the dataset in X
pca.fit(X)

#Eigenvalues (= squared singular values)
print('Eigenvalues:\n', pca.singular_values_**2)

#Eigenvectors (principal components, transposed)
print('Eigenvectors (signs may differ):\n', pca.components_.T)

# Explained variance
explained_ratio = pca.explained_variance_ratio_
print('Percentage of explained variance:', explained_ratio)

# Example of transformed data
T = pca.transform(X)
print('The same (transformed) data point as before:', T[21])


# we have trasformed the dimensions of the data, now we want to keep the first k colums
# how do I choose k?

# possible strategie:
# 1. Keep all the principal components until we cumulate at least 90% (or 95%) of the variance;
# 2. Keep all the principal components which explain more of the 5% of the variance;
# 3. Find the elbow in the cumulated variance.

explained = pca.explained_variance_
explained_variance = np.cumsum(explained) / sum(explained)
explained_variance = np.insert(explained_variance, 0, 0.)

plt.figure(figsize=(12,7))
plt.plot(range(5), explained_variance)
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.xticks(range(5))
plt.yticks(np.arange(0., 1.1, 0.1))
plt.grid()
plt.savefig("ml/plot/ms_fig2.png", dpi=150)

# Elbow = poin in the curve in which it stops raising and start to flatter

# we can see that the elbow is the cumulative variance is at 2 components,
# so we can choose k = 2

# without Scikit Learn
T_12 = np.dot(X_tilde, W[:,:2])

print('Data point with reduced dimensionality', T_12[21])

# with Scikit Learn
pca2 = PCA(n_components=2)
T_12 = pca2.fit_transform(X)

print('Data point with reduced dimensionality', T_12[21])

# the dataset is stored in the variable iris
y = iris.target.copy()
# 0 - setosa, 1 - versicolor, 2 - virginica
y[y == 1] = 2
y[y == 0] = 1
y[y == 2] = 0

print('Iris-Setosa targets\n', y)

# we consider logistic regression on two different datasets

# logistic regression on the original dataset
log_classifier_original = LogisticRegression(penalty=None) # regularization is applied as default
log_classifier_original.fit(X, y)
y_hat_original = log_classifier_original.predict(X)
acc_original = sum(y_hat_original == y) / len(y)
print('Accuracy original:', acc_original)

# logistic regression on the first two pc
T_12 = np.dot(X_tilde, W[:,:2])
log_classifier_12 = LogisticRegression(penalty=None)
log_classifier_12.fit(T_12, y)
y_hat_12 = log_classifier_12.predict(T_12)
acc_12 = sum(y_hat_12 == y) / len(y)
print('Accuracy first two pc:', acc_12)

# logiastic regression on the last two pc
T_34 = np.dot(X_tilde, W[:,2:])
log_classifier_34 = LogisticRegression(penalty=None)
log_classifier_34.fit(T_34, y)
y_hat_34 = log_classifier_34.predict(T_34)
acc_34 = sum(y_hat_34 == y) / len(y)
print('Accuracy last two pc:', acc_34)

# Principal component: New features obtained by combining the original ones

# Here:
# PC1 = 0.361·x̃₀ − 0.085·x̃₁ + 0.857·x̃₂ + 0.358·x̃₃ → explains 92.5% of the variance
# PC2 = −0.657·x̃₀ − 0.730·x̃₁ + 0.173·x̃₂ + 0.075·x̃₃ → explains 5.3%
# PC3 = −0.582·x̃₀ + 0.598·x̃₁ + 0.076·x̃₂ + 0.546·x̃₃ → explains 1.7%
# PC4 = 0.315·x̃₀ − 0.320·x̃₁ − 0.480·x̃₂ + 0.754·x̃₃ → explains 0.5%

# The percentage of how much they explain is obtained by the
# component’s eigenvalue divided by the sum of all eigenvalues

# Fig. 2 is the plot of the cumulativa of the measure of explanation of each principal component
# x: the number of principla component, y: the cumulative percetage of its explanation

# By priting the accuracy of model with all principl component, with first two pc and with last two pc we see:
# Accuracy original: 1.0
# Accuracy first two pc: 1.0
# Accuracy last two pc: 0.6666666666666666

# The first two pc are the most important as we can see in the plot og figure 2


# Now we visualize the data in the space defined by the first two principal components as opposed to the last two

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))

setosa = T_12[y == 1]
not_setosa = T_12[y == 0]

ax1.scatter(setosa[:, 0], setosa[:, 1], label='setosa')
ax1.scatter(not_setosa[:, 0], not_setosa[:, 1], label='not setosa', marker='x')
ax1.set(xlabel='pc1', ylabel='pc2')
ax1.grid()
ax1.legend()

setosa = T_34[y == 1]
not_setosa = T_34[y == 0]

ax2.scatter(setosa[:, 0], setosa[:, 1], label='setosa')
ax2.scatter(not_setosa[:, 0], not_setosa[:, 1], label='not setosa', marker='x')
ax2.set(xlabel='pc3', ylabel='pc4')
ax2.grid()
ax2.legend()

plt.savefig("ml/plot/ms_fig3.png", dpi=150)

# we can see that the data in the space defined by the two first pc are well diveded
# the data in the space defined by the two last pc instead are not well dived


# 3. Regularization

# We want to tune the model complexity through a hyper-paramter (commonly denoted as  λ)

# We want to consider regularization technique in the context of a classification task
# with the k-Nearest Neighbors (k-NN) classifier. k-NN is a non-parametric method that coarsely
# works as follows: Suppose you have to predict the class of a new data point,
# look at the  k  data points in the train set that are the closest to this new one,
# and decides the class of it accordingly (usually via majority voting).

# As it is common in non-parametric methods, the k-NN classifier does not involve a real training phase,
# it just has to keep in memory the train set to make predictions.
# However, we need to properly choose the hyper-parameter  k  beforehand,
# which is actually a regularization parameter.

iris = datasets.load_iris()

# the variable iris now contains inputs ('data' field), targets ('target', field)
# and other stuff. Let us keep the inputs and targets in two separate variables

X = iris.data
y = iris.target

print(iris.feature_names)

# set the seed to get consistent splits
np.random.seed(0)

# we update the targets according to our preferred task
# 0 - setosa, 1 - versicolor, 2 - virginica
y[y == 2] = 0

# let us concatenate X and y in a single matrix
D = np.concatenate((X[:, :2], y.reshape(len(y), -1)), axis=1)

# then we shuffle D
np.random.shuffle(D)

# and we slip the data as 50/20/20
X_train = D[:100,:2]
y_train = D[:100,2]
X_test = D[100:150,:2]
y_test = D[100:150,2]

versicolor = X_train[y_train == 1]
not_versicolor = X_train[y_train == 0]

plt.figure(figsize=(12,7))
plt.scatter(versicolor[:, 0], versicolor[:, 1], label='versicolor')
plt.scatter(not_versicolor[:, 0], not_versicolor[:, 1], label='not versicolor', marker='x')

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.grid()
plt.legend()
plt.savefig("ml/plot/ms_fig4.png", dpi=150)

# function to plot the decision surface of the k-NN classifier,
# see https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py

def knn_decision_surface(ax, X, clf, k):
  h = .02
  x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
  y_min, y_max = X[:, 1].min() - .2, X[:, 1].max() + .2
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  cmap_light = ListedColormap(['orange', 'cornflowerblue'])

  ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.4)
  ax.scatter(versicolor[:, 0], versicolor[:, 1], label='versicolor', s=15)
  ax.scatter(not_versicolor[:, 0], not_versicolor[:, 1], label='not versicolor', marker='x', s=15)
  ax.set_title(f"k = {k}")
  ax.set_xlabel('sepal length')
  ax.set_ylabel('sepal width')

fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharex=True, sharey=True)

for k, ax in zip(np.arange(1, 11), axes.flatten()):
  knn_classifier = neighbors.KNeighborsClassifier(k)
  knn_classifier.fit(X_train[:, :2], y_train)
  knn_decision_surface(ax, X, knn_classifier, k)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.tight_layout()
plt.savefig("ml/plot/ms_fig5.png", dpi=150)


# k-NN

# k-Nearest Neighbors classifies a new point like this:

# 1. find the k training points closest to the new poit you want to classify (by Euclidean distance),
# 2. assign the new point to class in which the majority of the k point blong.

# So there is not training and prediction, there is only prediction.
# It's called non-parametric: there's no vector of weights w to learn. The only thing to choose is k.

# How do I choode k?

# Choose it like any other hyperparameter: try several values and keep the one with the lowest error
# on data the model didn't see, using cross-validation

# From figure 4 I can make qualitative observations about the shape of the model.
# (It's not enough to choose k)

parameters = np.arange(1, 11)
train_accuracy = []
cv_accuracy = []
test_accuracy = []

for k in parameters:
  knn_classifier = neighbors.KNeighborsClassifier(k)
  knn_classifier.fit(X_train[:,:2], y_train)

  y_hat = knn_classifier.predict(X_train)
  accuracy = sum(y_hat == y_train) / len(y_train)
  train_accuracy.append(accuracy)

  cv_accuracy.append(cross_val_score(neighbors.KNeighborsClassifier(k), X_train, y_train, cv=5).mean())

  y_hat = knn_classifier.predict(X_test)
  accuracy = sum(y_hat == y_test) / len(y_test)
  test_accuracy.append(accuracy)


plt.figure(figsize=(12, 7))
plt.plot(parameters, train_accuracy, label='train accuracy')
plt.plot(parameters, cv_accuracy, label='cv accuracy')
plt.plot(parameters, test_accuracy, label='test accuracy')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.ylim([0, 1])
plt.legend()
plt.savefig("ml/plot/ms_fig6.png", dpi=150)

best_k = parameters[np.argmax(cv_accuracy)]
best_knn = neighbors.KNeighborsClassifier(best_k).fit(X_train, y_train)
print('Best k (cross-validation):', best_k)
print('Test accuracy with best k:', best_knn.score(X_test, y_test))

# We creat k k-NN classifier
# For each k we predict the training set and compute the training accurancy
# For each k we predict the test set and compute the test accurancy

# We have a bias-variance trade-off once again."

# Small k → high variance (overfitting): the model fits the training set very well (low training error
#  but does worse on the test set (higher test error). This matches the irregular islands in the figures.
# Larger k → lower variance: training accuracy goes down and test accuracy goes up.
# The model generalizes better, matching the smoother surfaces.

# For choosing k we choose the value associated with the higher test accurancy, so k ≈ 7


# RECAP:
# Strategies to improve our model:

# 1. Feature selection:
# Choose a subset of the original features and discard the rest, keep only the features that
# actually help predict the output.
# Backward elimination: start with all features. Remove one at a time, each time dropping the feature
# whose removal doesn't lower validation accuracy. Stop when every remaining removal hurts.
# Forward selection does the opposite: start with none and add one at a time.

# 2. PCA (dimensionality reduction):
# Replace the original features with a few new features (the principal components),
# each a linear combination of all the original ones.
# Center the data, compute the covariance matrix, take its eigenvectors (directions)
# and eigenvalues (variance along each one), sort them by variance, keep the first k,
# and project the data onto them. Choose k with the elbow or a variance threshold (for example 90–95%).
# Note: it's unsupervised, so it doesn't use the labels.

# 3. Regularization:
# Keep all the features, but use a hyperparameter that limits how complex the model can be.
# In ridge/lasso, a penalty λ shrinks the weights.
# In k-NN, k plays this role: a larger k averages over more neighbors and gives a smoother boundary.
# Choose the value with validation or cross-validation accuracy.