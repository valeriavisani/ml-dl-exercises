import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from scipy.stats import zscore
from sklearn import linear_model
from statsmodels import api as sm

# python -m ml.linear_regression

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = iris_dataset = pd.read_csv(url, names=names)

print(dataset.head())

# We want to predict the "petal width" given the "petal length" for samples in
# the class "Iris Setosa"

# Supervised Learning, we have data: x and y(x) (the output of y(x))
# Linear Regression, hypothesis function h(w, x), w are parameters, h is linear in w
# We want h(w, x) return something as close as we can to  y(x)

# x value, input
# y(x) value, output of y(x), output that I want to obtain with h(h, w), "label"
# "Samples" are data, dataset row (one row one sample)
# "Features" are characteristics of data, dataset column

# Loss function: we use "Residual sum of squares"

dataset.describe()

dataset.hist(figsize=(16,9))
plt.savefig("ml/plot/lr_fig1.png", dpi=150)
plt.close()

scatter_matrix(dataset, figsize=(16, 9))
plt.savefig("ml/plot/lr_fig2.png", dpi=150)
plt.close()

# x axis: Petal-length - y axis: Petal-width
dataset.plot.scatter('petal-length', 'petal-width', grid=True, figsize=(16,9))
plt.savefig("ml/plot/lr_fig3.png", dpi=150)
plt.close()

# 1. Load data
# 2. Inspect data
# 3. Select interesting data
# 4. Preprocessing
#    shuffing, remove incosistent data, remove outliers, normalize data,
#    fill missing data

# Data Normalization, examples:
# 1. z-score: Trasform data of each sample for each in such a way to have
#             mean = 0 and standard derivation = 1
# 2. Scaling in the range 0-1

# data

#input
x = zscore(dataset['petal-length'].values).reshape(-1, 1)
# we reshape our feature column as a (n_sample, n_features) matrix
# target
y = zscore(dataset['petal-width'].values)

# linear model

lin_model = linear_model.LinearRegression()
lin_model.fit(x, y)


w1 = lin_model.coef_ # weights of the model are stored here
w0 = lin_model.intercept_ # and here it is the intercept
# (first value of w that isn't a coefficient)

# Compute the y component of the regression line

y_pred = lin_model.predict(x)

plt.figure(figsize=(16,9))
plt.scatter(x, y)
plt.plot(x, y_pred, label='predicted', color='red')
plt.savefig("ml/plot/lr_fig4.png", dpi=150)
plt.close()

lin_model2 = sm.OLS(y, x).fit()
print(lin_model2.summary())

