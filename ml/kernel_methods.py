import numpy as np
from sklearn import svm
from sklearn import datasets
from scipy.stats import zscore
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Here we will study kernel methods.
# First use kernel methods in Gaussian Process Regression for (non-linear) regression
# and then with Support Vector Machine for classification (linear or, with a kernel, non-linear)


# Kernel Methods:
# If I have to classify two classes but the line that dived this two classes is not linear
# how can I divided them with linear methods?
# Kernel methods: move data to a space with higher dimension in such a way that data can be
# diveded with a linear model
# Es.:
# xa = (x1, x2)
# phi -> feature map
# phi(xa) = (x1^2, x2^2, sqrt(2)x1x2)
# (here example of classification, but note that kernel methods can be apply both to classification
# and to regression)

# Kernel Trick:
# If I need only the dot product of data, <xa,xb>,*
# I consequenly need only <phi(xa),phi(xb)>
# Kernel trick: I can compute direclty <phi(xa),phi(xb)> without compute
# separatly phi(xa) and phi(xb), like this is less expansive
# K(xa, xb) = <phi(xa),phi(xb)>, the kernel is this fuction

# *If it's not like this many algorithms (ridge regression, SVM, perceptron, PCA)
# can be rewritten in such a way that data appear only as a dot product, it's called dual form



# 1. Gaussian Process Regression


# Gaussian Process regression:
# Data: inputs x1,...,xN and targets t1,...,tN
# Function I want to predict: y(x), observed with noise: tn = y(xn) + eps, eps ~ N(0, sigma^2)
#
# PRIOR (before seeing the data, before fit):
#   I assume that the values y(x1),...,y(xN) have a joint Gaussian distribution:
#   (y(x1),...,y(xN)) ~ N( 0, K ),  K_nm = k(xn, xm)
#   the kernel k is the covariance: it says how correlated the values at two
#   points are, so it controls how smooth the function is (length scale l)
#
# POSTERIOR (after seeing the data, after fit):
#   After the fit, for a new point x I obtain a prediction m(x), with uncertainty σ²(x)
#   y(x) ~ N( m(x), sigma^2(x) )
#   m(x)       = prediction
#   sigma^2(x) = uncertainty:
#                small: the model is confident (x is near the training inputs)
#                large: the model is not confident (x is far from the training inputs)
#   The uncertainty is related to the fact that a new point, x, a new point from a new dataset,
#   is close or far from point of the trainig dataset
#
# So the GP gives both a prediction and its uncertainty


dataset = datasets.load_iris()

x = zscore(dataset.data[:, 2]).reshape(-1, 1) # column 2 of data is petal length
t = zscore(dataset.data[:, 3]) # column 3 of data is petal width

# Define combinations of l and sigma_sq to test
# l = kernel length scale, sigma_sq = noise variance on the targets
L_values = [0.1, 0.1, 10, 10]
SIGMA_SQ_values = [0.1, 1, 0.1, 1]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Gaussian Process Regression for Different Parameters', fontsize=22)

for i, (l, sigma_sq) in enumerate(zip(L_values, SIGMA_SQ_values)):
    print(f"Testing combination {i+1}: l={l}, sigma_sq={sigma_sq}")

    # Determine subplot position
    row = i // 2
    col = i % 2
    ax = axes[row, col]

    # Kernel and GP parameters
    kernel = RBF(l, length_scale_bounds="fixed")
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=sigma_sq).fit(x, t)

    x_pred = np.linspace(np.min(x),np.max(x),5000).reshape(-1,1)
    t_pred, sigma = gpr.predict(x_pred, return_std=True)

    ax.plot(x, t, 'r.', markersize=10, label='Observations')
    ax.plot(x_pred, t_pred, 'b-', label='Prediction')
    ax.fill(np.concatenate([x_pred, x_pred[::-1]]),
             np.concatenate([t_pred - sigma,
                            (t_pred + sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='±1 std')
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$t$', fontsize=14)
    ax.set_title(f'l={l}, $\\sigma^2$={sigma_sq}', fontsize=16)
    ax.legend(loc='upper left', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig("ml/plot/km_fig1", dpi=150)


# l controls complexity: small l gives a local, wiggly fit, while large l gives a global, nearly linear one.
# σ² controls how far the data is trusted: a large σ² means more smoothing,
# more shrinkage toward 0 and wider uncertainty.


# Support Vector Machine - SVM

# Data preparation

input = zscore(dataset.data[:, [0, 1]]) # column 0,1 are sepal length and sepal width

target = dataset.target.copy()
# 0 - setosa, 1 - versicolor, 2 - virginica
target[target == 1] = 2 # versicolor becomes a (different) non-target class (2)
target[target == 0] = 1 # setosa becomes target class (1)
target[target == 2] = 0 # virginica and versicolor becomes a non-target class (0)

for i in range(2):

    if i==0:
        SVM_model = svm.SVC(kernel="linear") # k(x, z) = x·z  -> straight boundary
    elif i==1:
        SVM_model = svm.SVC() # default kernel = 'rbf' (Gaussian) -> curved boundary
    else:
        raise NotImplementedError()

    SVM_model.fit(input, target)

    plt.figure(figsize=(12,7))

    plt.scatter(input[:, 0], input[:, 1], c=target, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    #evaluate the SVM value for the positive class
    Z = SVM_model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary (f(x) = 0) and margins (f(x) = 1 and f(x) = -1)
    # (for the linear kernel f(x) = w^Tx + b; for RBF f(x) = sum_n a_n t_n k(x_n, x) + b)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
              linestyles=['--', '-', '--'])
    ax.scatter(SVM_model.support_vectors_[:, 0], SVM_model.support_vectors_[:, 1],
              s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.savefig(f"ml/plot/km_fig2{'a' if i == 0 else 'b'}", dpi=150)

# Setosa vs not setosa is almost linearly separable with these features,
# so the linear kernel is enough and preferable: simpler boundary, fewer support vectors,
# and by the LOO bound (error ≤ #SV / N) a better generalization guarantee: 12/150 = 8% vs 23/150 ≈ 15%.
# The RBF kernel adds flexibility this problem doesn't need. On harder problems
# (like versicolor vs not versicolor in the k-NN exercise), the RBF kernel would be the one that helps.
# In both cases, C (and for RBF also the kernel width gamma) should be chosen with cross-validation.

# Homework

# Prediction petal width using all the other features as an input

dataset = datasets.load_iris()

x = zscore(dataset.data[:, [0, 1, 2]])  # sepal length, sepal width, petal length

t = zscore(dataset.data[:, 3]) # petal width (target)

np.random.seed(0)
D = zscore(dataset.data)
np.random.shuffle(D)

# split 70/30
# (150 rows * 0,70 = 105)
# 70
X_train = D[:105, :3]
t_train = D[:105, 3]
# 30
X_test = D[105:, :3]
t_test = D[105:, 3]

kernel = RBF(length_scale=[1.0, 1.0, 1.0])
# length scale is a vetor that for each input feature associate a number, li
# importance of the feature for the prediction move like 1/li
# small li -> i-th feature is important

gpr = GaussianProcessRegressor(kernel=kernel, alpha = 0.1)
# alpha = 0.1: variance sigma^2 of the noise on the targets,
# tn = y(xn) + eps, with eps ~ N(0, alpha)
# - small alpha (e.g. 1e-10, the default): the GP fully trusts the data
#   and passes through every point -> overfitting
# - large alpha (e.g. 1): the GP considers the data very noisy
#   -> smoother curve, risk of underfitting

gpr.fit(X_train, t_train)

t_pred, sigma = gpr.predict(X_test, return_std=True) 


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

x_plot1 = X_test[:, 2]                 # petal length (test set) on the x-axis
order = np.argsort(x_plot1)            # indices that sort x from smallest to largest

ax1.scatter(x_plot1, t_test, label='True target')                  # test targets as points
# dots: x: x_plt, y: t_test
ax1.plot(x_plot1[order], t_pred[order], 'r-', label='Prediction')  # predictions as a line
# line: x: x_plt, y: t_pred
ax1.set_xlabel('Petal length')
ax1.set_ylabel('Petal width')
ax1.legend()


x_plot2 = X_test[:, 0]                 # sepal length (test set) on the x-axis
order = np.argsort(x_plot2)            # indices that sort x from smallest to largest

ax2.scatter(x_plot2, t_test, label='True target')                  # test targets as points
# dots: x: x_plt, y: t_test
ax2.plot(x_plot2[order], t_pred[order], 'r-', label='Prediction')  # predictions as a line
# line: x: x_plt, y: t_pred
ax2.set_xlabel('Sepal length')
ax2.set_ylabel('Petal width')
ax2.legend()

x_plot3 = X_test[:, 1]                 # sepal width (test set) on the x-axis
order = np.argsort(x_plot3)            # indices that sort x from smallest to largest

ax3.scatter(x_plot3, t_test, label='True target')                  # test targets as points
# dots: x: x_plt, y: t_test
ax3.plot(x_plot3[order], t_pred[order], 'r-', label='Prediction')  # predictions as a line
# line: x: x_plt, y: t_pred
ax3.set_xlabel('Sepal width')
ax3.set_ylabel('Petal width')
ax3.legend()


plt.savefig("ml/plot/km_fig3.png", dpi=150)

# Petal width vs petal length: strong, almost linear correlation
# Petal width vs sepal length: weak positive correlation
# Petal width vs sepal width: no clear correlation (slightly negative)

# Positive correlation: when one grows, the other tends to grow too. Points go up from left to right. (Petal length ↑ → petal width ↑.)
# Negative correlation: when one grows, the other tends to decrease. Points go down from left to right.
# No correlation: no clear direction; the points are scattered.


plt.figure(figsize=(8, 8))
plt.scatter(t_test, t_pred)
# x = true value, y = predicted value
lims = [t_test.min(), t_test.max()]
plt.plot(lims, lims, 'k--', label='Perfect prediction')   # diagonal y = x
plt.xlabel('True petal width')
plt.ylabel('Predicted petal width')
plt.legend()
plt.savefig("ml/plot/km_fig4.png", dpi=150)

# x = true value, y = predicted value
# If predition value = true value --> y = x, diagonal

mse = mean_squared_error(t_test, t_pred)    # mean of (true - pred)^2, lower is better
rmse = np.sqrt(mse)                         # same units as the target
mae = mean_absolute_error(t_test, t_pred)   # mean of |true - pred|
r2 = r2_score(t_test, t_pred)               # 1 = perfect, 0 = as good as predicting the mean

print('MSE:', mse)
print('RMSE:', rmse)
print('MAE:', mae)
print('R^2:', r2)

# R² = 0.95: it explains about 95% of the variability in petal width.
# RMSE ≈ 0.22 and MAE ≈ 0.16 are in z-score units (standard deviations of petal width).
# Since one standard deviation is about 0.76 cm, the typical error is about 0.12–0.17 cm.
# MSE ≈ 0.048: much lower than ~1, which is what you'd get by always predicting the mean.


# Now I want to use SVM for classify setosa and not setosa

dataset = datasets.load_iris()

x = zscore(dataset.data[:, [0, 1, 2, 3]])  # sepal length, sepal width, petal length, petal width
# (all 4 features, I can do: x = zscore(dataset.data))

target = dataset.target.copy()
# 0 - setosa, 1 - versicolor, 2 - virginica
target[target == 1] = 2 # versicolor becomes a (temporary) non-target class (2)
target[target == 0] = 1 # setosa becomes target class (1)
target[target == 2] = 0 # versicolor and virginica become non-target class (0)


np.random.seed(0)
D = np.concatenate((x, target.reshape(-1, 1)), axis=1) 
np.random.shuffle(D)

# split 70/30
# (150 rows * 0,70 = 105)
# 70
X_train = D[:105, :4]
t_train = D[:105, 4]
# 30
X_test = D[105:, :4]
t_test = D[105:, 4]


SVM_model = svm.SVC()

SVM_model.fit(X_train, t_train)

t_pred = SVM_model.predict(X_test) 

# Metrics:

print('Accuracy:', accuracy_score(t_test, t_pred))     # fraction of correct predictions
print('Precision:', precision_score(t_test, t_pred))   # of the predicted setosa, how many are really setosa
print('Recall:', recall_score(t_test, t_pred))         # of the real setosa, how many were found
print('F1:', f1_score(t_test, t_pred))                 # balance of precision and recall
print('Confusion matrix:\n', confusion_matrix(t_test, t_pred))   # [[TN, FP], [FN, TP]]

# All metrics 1, setosa and not setosa are linearly separable



# Now I want to use SVM for classify versicolar and not versicolar

dataset = datasets.load_iris()

x = zscore(dataset.data[:, [0, 1, 2, 3]])  # sepal length, sepal width, petal length, petal width
# (all 4 features, I can do: x = zscore(dataset.data))

target = dataset.target.copy()
# 0 - setosa, 1 - versicolor, 2 - virginica
target[target == 2] = 0 # virginica and setosa become non-target class (0)
# (versiocolar remain 1, target class)


np.random.seed(0)
D = np.concatenate((x, target.reshape(-1, 1)), axis=1) 
np.random.shuffle(D)

# split 70/30
# (150 rows * 0,70 = 105)
# 70
X_train = D[:105, :4]
t_train = D[:105, 4]
# 30
X_test = D[105:, :4]
t_test = D[105:, 4]


SVM_model = svm.SVC()

SVM_model.fit(X_train, t_train)

t_pred = SVM_model.predict(X_test) 

# Metrics:

print('Accuracy:', accuracy_score(t_test, t_pred))     # fraction of correct predictions
print('Precision:', precision_score(t_test, t_pred))   # of the predicted versicolor, how many are really versicolor
print('Recall:', recall_score(t_test, t_pred))         # of the real versicolor, how many were found
print('F1:', f1_score(t_test, t_pred))                 # balance of precision and recall
print('Confusion matrix:\n', confusion_matrix(t_test, t_pred))   # [[TN, FP], [FN, TP]]


# Accuracy: 0.9555555555555556
# Precision: 1.0
# Recall: 0.875
# F1: 0.9333333333333333

# Here prediction still good but a little worse