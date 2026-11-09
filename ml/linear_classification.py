import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.utils import shuffle
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# python -m ml.linear_classification

# Supervised Learning
# regression --> target continuos
# classification --> target discrete

# Approaches for classification:
# 1. Discriminant Fuction Approach:
#    Model a function that maps an input to a class
# 2. Probabilistic Discriminative Approach:
#    Model the conditional probability given an input of being in a specif class, p(Ck|x)
# 3. Probabilistic Generative Approach
#    Model the lilelihood p(x|Ck) and the prior p(Ck)

# Binary classification (2 classes)

# We want to predict the kind of Iris (setosa or not setosa) given sepal/petal length and width
# We decide to use speal length and sepal width (to visualize)

# Discriminant Fuction Approach

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = iris_dataset = pd.read_csv(url, names=names)
print(dataset.head())

# Data

# input: (normalised)
X = zscore(dataset[['sepal-length', 'sepal-width']].values) # Normalize
# target:
t = (dataset['class'].values == 'Iris-setosa')
X, t = shuffle(X, t, random_state=0) # (shuffle rows of data because the dataset was sortd)

plt.figure(figsize=(12,7))
setosa = X[t]
not_setosa = X[~t]
plt.scatter(setosa[:, 0], setosa[:, 1], label='setosa')
plt.scatter(not_setosa[:, 0], not_setosa[:, 1], label='not setosa', marker='x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.legend()
plt.savefig("ml/plot/lc_fig1.png", dpi=150)


# First classifier: the Dummy Classifier
# it assign for each input variable the most frequent lables
# baseline, if a classifier perform wrost than the dubby classifier is useless

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X, t)
most_frequent = dummy.predict(X)

# Evaluate classification

# We can evoluate classification with the "Confusion Matrix"
# given the infotmation of real classification and predicct classification
# it contain the number of: True Positive, False Positive, True Negative, False Negative

# Given: tp, fp, tn, fn and N (number of samples) is possible to compute:
# Accurancy (Acc): (tp + tn)/N
# Precision (Pre): tp/(tp + fp)
# Recall (Rec): tp/(tp + fn)
# F1 Score: (2*Pre*Rec)/(Pre + Rec)

print(f"Accuracy: {accuracy_score(t, most_frequent)}\n"
      f"Precision: {precision_score(t, most_frequent)}\n"
      f"Recall: {recall_score(t, most_frequent)}\n"
      f"F1: {f1_score(t, most_frequent)}\n"
      f"Confusion Matrix:\n{confusion_matrix(t, most_frequent)}")


# Second classifier: the Percepton Classifier

# Hypothesis space:
# y(x_n) = sign(w^T x_n) = sign(w_0 + x_{n1}*w_1 + x_{n2}*w_2)

# Loss function:
# Penalizes misclassified points proportionally to their distance
# from the separating hyperplane.
# L_P(w) = - sum_{n in M} (w^T x_n) * C_n

# Optimization method:
# Online Gradient Descent (OGD)

perc_classifier = Perceptron(shuffle=False, random_state=0)
perc_classifier.fit(X, t)

print(perc_classifier.coef_)

plt.figure(figsize=(12,7))
plt.scatter(setosa[:, 0], setosa[:, 1], label='setosa')
plt.scatter(not_setosa[:, 0], not_setosa[:, 1], label='not setosa', marker='x')

coef = perc_classifier.coef_.flatten() # weights
w0 = perc_classifier.intercept_ # bias
w1 = coef[0]
w2 = coef[1]

step = 100
ds_x1 = np.linspace(X[:,0].min(), X[:,0].max(), step)
# Compute x2 componenent given some x1:
# w^T x + x0 = 0 -> w0 + w1 * x1 + w2 * x2 = 0 -> x2 = - (w0 + w1*x1) / w2
ds_x2 = [-(w0 + w1*x1) / w2 for x1 in ds_x1]
plt.plot(ds_x1, ds_x2, label='DS')

plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.legend()
plt.savefig("ml/plot/lc_fig2.png", dpi=150)


t_pred = perc_classifier.predict(X)

print(f"Accuracy: {accuracy_score(t, t_pred)}\n"
      f"Precision: {precision_score(t, t_pred)}\n"
      f"Recall: {recall_score(t, t_pred)}\n"
      f"F1: {f1_score(t, t_pred)}\n"
      f"Confusion Matrix:\n{confusion_matrix(t, t_pred)}")


# Implementing percepton classifier

w = np.ones(3)
n_epochs = 10
for epoch in range(n_epochs):
  for i, (x_i, t_i) in enumerate(zip(X, t)):
    # correct t_i to be in {-1, 1}
    corr_t_i = 1 if t_i else -1
    ext_x = np.concatenate([np.ones(1), x_i.flatten()])
    if np.sign(w.dot(ext_x)) != corr_t_i:
      w = w + ext_x * corr_t_i

def plot_ds(X, w, step=100, label='DS'):
  ds_x1 = np.linspace(X[:,0].min(), X[:,0].max(), step)
  ds_x2 = [-(w[0] + w[1]*x1) / w[2] for x1 in ds_x1]
  plt.plot(ds_x1, ds_x2, label=label)

plt.figure(figsize=(12,7))
plt.scatter(setosa[:, 0], setosa[:, 1], label='setosa')
plt.scatter(not_setosa[:, 0], not_setosa[:, 1], label='not setosa', marker='x')

# Implemented Perceptron
plot_ds(X, w, label='Implemented Perceptron DS')

# Sklearn Perceptron
coef = perc_classifier.coef_.flatten() # weights
w0 = perc_classifier.intercept_ # bias

perc_w = np.array([w0.item(), coef[0], coef[1]])
plot_ds(X, perc_w, label='Scikit-Learn Perceptron DS')

plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.legend()
plt.savefig("ml/plot/lc_fig3.png", dpi=150)

# Probabilistic Discriminative Approach

# Classifier: Logic Regression

# Hypothesis space:
# Predicts the probability of belonging to class 1
# using the sigmoid activation function.
# y_n = sigma(w_0 + x_n1*w_1 + x_n2*w_2)

# Loss function:
# Binary Cross-Entropy (Negative Log-Likelihood).
# Measures the discrepancy between predicted probabilities
# and the true class labels.
# L(w) = -sum_{n=1}^N [C_n*ln(y_n) + (1-C_n)*ln(1-y_n)]

# Optimization method:
# Gradient Descent

# Sigmoid function:
# sigma(x) = 1 / (1 + exp(-x))


log_classifier = LogisticRegression(penalty=None) # default is penalty='l2', that is l2 regularization!
log_classifier.fit(X, t)


# compare perceptron, by hand and logistic regression
plt.figure(figsize=(12,7))
plt.scatter(setosa[:, 0], setosa[:, 1], label='setosa')
plt.scatter(not_setosa[:, 0], not_setosa[:, 1], label='not setosa', marker='x')

# Implemented Perceptron
plot_ds(X, w, label='Implemented Perceptron DS')

# Sklearn Perceptron
coef = perc_classifier.coef_.flatten() # weights
w0 = perc_classifier.intercept_ # bias
perc_w = np.array([w0.item(), coef[0], coef[1]])
plot_ds(X, perc_w, label='Scikit-Learn Perceptron DS')

# Sklearn Logistic Regression
coef = log_classifier.coef_.flatten() # weights
w0 = log_classifier.intercept_ # bias
log_w = np.array([w0.item(), coef[0], coef[1]])
plot_ds(X, log_w, label='Scikit-Learn Logistic Regression DS')

plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.legend()
plt.savefig("ml/plot/lc_fig4.png", dpi=150)


# Multiple classes

multi_t = shuffle(dataset['class'].values, random_state=0)   # target with 3 classes: Setosa, Versicolor, Virginica
multi_log_classifier = LogisticRegression()    # same model
multi_log_classifier.fit(X, multi_t)           # training

# Probabilistic Generative Approach
# (model the lilelihood p(x|Ck) and the prior p(Ck))

# Naive Bayes Classifier

# Hypothesis space:
# Predicts the class with the highest posterior probability,
# assuming conditional independence among features.
# y_n = argmax_k [P(C_k) * product_{j=1}^M P(x_j | C_k)]

# Loss function:
# Log-Likelihood
# Evaluates how well the model parameters explain the observed data.

# Optimization method:
# Maximum Likelihood Estimation (MLE)
# Estimates the model parameters by maximizing the likelihood
# of the training data.

from sklearn.naive_bayes import GaussianNB

gnb_classifier = GaussianNB()
gnb_classifier.fit(X, t)
t_pred = gnb_classifier.predict(X)

print(f"Accuracy: {accuracy_score(t, t_pred)}\n"
      f"Precision: {precision_score(t, t_pred)}\n"
      f"Recall: {recall_score(t, t_pred)}\n"
      f"Confusion Matrix:\n{confusion_matrix(t, t_pred)}")

# Generative approach, I can use it to generate new data

# Generate new data:

N = 100

new_samples = np.empty((N, 2))
new_t = np.empty(N, dtype=bool)

for i in range(N):
  # Based on the class priors, we sample a class
  class_ = np.random.choice([0,1], p=gnb_classifier.class_prior_)
  new_t[i] = class_

  # For each feature, we have a normal distribution of its likelihood given the class

  # theta: mean of each feature per class (n_classes, n_features)
  thetas = gnb_classifier.theta_[class_, :]

  # sigma: variance of each feature per class (n_classes, n_features)
  sigmas = gnb_classifier.var_[class_, :]

  # sample x1
  new_samples[i,0] = np.random.normal(thetas[0], sigmas[0], 1)
  # sample x2
  new_samples[i,1] = np.random.normal(thetas[1], sigmas[1], 1)

# divide samples by class
new_setosa = new_samples[new_t, :]
new_not_setosa = new_samples[~new_t, :]


plt.figure(figsize=(12,7))

# plot real samples
plt.scatter(setosa[:, 0], setosa[:, 1], label='setosa', color='red')
plt.scatter(not_setosa[:, 0], not_setosa[:, 1], label='not setosa', color='blue')

# plot generated samples
plt.scatter(new_setosa[:, 0], new_setosa[:, 1], label='new, setosa', color='red', marker='x', alpha=0.3)
plt.scatter(new_not_setosa[:, 0], new_not_setosa[:, 1], label='new, not setosa', color='blue', marker='x', alpha=0.3)

plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.legend()
plt.savefig("ml/plot/lc_fig5.png", dpi=150)
