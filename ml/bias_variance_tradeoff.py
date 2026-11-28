import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)

# E[(t-y(x))^2] = σ^2 + Var[y(x)] + (f(x)-E[y(x)])^2
# Error = noise + variance + bias^2
# You can't remove noise
# You have to find a balance between variance and bias

n_points = 1000
eps = 0.7
 
def fun(x):
  return 1 + 1/2 * x + 1/10 * x**2

x = np.random.uniform(low=0, high=5, size=(n_points, 1))

t = fun(x)

t_noisy = t + eps * np.random.randn(n_points, 1)

# We generate a dataset, input x, output t_noist
# This time we know f(x), so we have a ground-truth about the model

phi = np.concatenate([x, x**2], axis=1)

# We modify the input with basis funciton, phi are the modified input

lin_model = linear_model.LinearRegression()
lin_model.fit(x, t_noisy)

qua_model = linear_model.LinearRegression()
qua_model.fit(phi, t_noisy)

# We defin two model of linear regression, one with x and one with phi

plt.figure(figsize=(16,9))
plt.scatter(x, t_noisy, label='True Points')
plt.scatter(x, lin_model.predict(x), color='red', label="Linear Model")
plt.scatter(x, qua_model.predict(phi), color='orange', label="Quadratic Model")

plt.legend()
plt.savefig("ml/plot/bv_fig1.png", dpi=150)

# We plot a figure with true point, prediction of lin_model, prediction of qua_model

# Real and best paramseters
real_par = np.array([1, 1/2, 1/10])
best_lin_par = np.array([7/12, 1, 0])

# Fitted parameters
lin_c = np.array([lin_model.intercept_[0], lin_model.coef_[0][0], 0])
qua_c = np.concatenate([qua_model.intercept_, qua_model.coef_.flatten()])

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
s = 70

ax.scatter(*real_par, marker='x', color='blue', s=s, label='Best Quadratic')
# This is the true funciton
ax.scatter(*best_lin_par, marker='o', color='red', s=s, label='Best Linear')
# best prediction you can have with linear model
ax.scatter(*qua_c, marker='+', color='blue', s=s, label='Fitted Quadratic')
# prediction with quadratic model
ax.scatter(*lin_c, marker='+', color='red', s=s, label='Fitted Linear')
# prediction with linear model

ax.set_xlabel('w_0')
ax.set_ylabel('w_1')
ax.set_zlabel('w_2')
ax.view_init(elev=20., azim=32)
plt.title('Parameter Space')
plt.legend()
plt.grid()
plt.savefig("ml/plot/bv_fig2.png", dpi=150)

# true function in quadratic, linear model can never reach exactly the true

# Now we repeat this pice of code more time and we plot each of them togethere to visualise better
# the concept of variace and bias: (then we will understande how to deal with the tradeoff)


# We write in a function the code we have seen above
def sample_and_fit(n_points):
  x = np.random.uniform(low=0, high=5, size=(n_points, 1))
  t = fun(x)
  t_noisy = t + eps * np.random.randn(n_points, 1)
  lin_model = linear_model.LinearRegression()
  lin_model.fit(x, t_noisy)
  phi = np.concatenate([x, x**2], axis=1)
  qua_model = linear_model.LinearRegression()
  qua_model.fit(phi, t_noisy)
  return lin_model, qua_model

# repeats the code above for n_repetition times, and saves the results
def multiple_sample_and_fit(n_repetitions, n_points):
  lin_coeff = np.zeros((n_repetitions, 3))
  qua_coeff = np.zeros((n_repetitions, 3))

  for i in range(n_repetitions):
    lin_model, qua_model = sample_and_fit(n_points)
    # Store the results
    lin_coeff[i, :] = np.concatenate([lin_model.intercept_, lin_model.coef_.flatten(), np.zeros(1)])
    qua_coeff[i, :] = np.concatenate([qua_model.intercept_, qua_model.coef_.flatten()])
  return lin_coeff, qua_coeff

# plot the models coefficients in a 3D space
def plot_models(lin_coeff, qua_coeff):
  fig = plt.figure(figsize=(16, 9))
  ax = fig.add_subplot(111, projection='3d')

  # (only these 2 lines have changed)
  ax.scatter(qua_coeff[:, 0], qua_coeff[:, 1], qua_coeff[:, 2], marker='.', color='blue', s=s, alpha=0.3)
  ax.scatter(lin_coeff[:, 0], lin_coeff[:, 1], lin_coeff[:, 2], marker='.', color='red', s=s, alpha=0.3)

  ax.scatter(*real_par, marker='x', color='blue', s=100)
  ax.scatter(*best_lin_par, marker='o', color='red', s=100)
  ax.set_xlabel('w_0')
  ax.set_ylabel('w_1')
  ax.set_zlabel('w_2')
  ax.view_init(elev=20., azim=32)
  plt.title('Parameter Space')
  plt.grid()
  plt.savefig("ml/plot/bv_fig3.png", dpi=150)


n_repetitions = 100
n_points = 1000
lin_coeff, qua_coeff = multiple_sample_and_fit(n_repetitions, n_points)
plot_models(lin_coeff, qua_coeff)

# We can see that the quadratic model (blue) ha no bias but has a big variace
# Instead the linear model (red) has a big bias but no variance


# Now we want to understand how to deal with the variance bias trade off

# We generate a new dataset
n_tot = 40
eps = 2

def fun(x):
  return (0.5 - x) * (5 - x) * (x - 3)

x = np.random.uniform(low=0, high=5, size=(n_tot, 1))
y = fun(x) + eps * np.random.randn(n_tot, 1)

# We divide the dataset into:
# - Training set, i.e., the data we will use to learn the model parameters
# - Validation set, i.e., the data we will use to select the model
# - Test set, i.e., the data we will use to evaluate the performance of our model
# (Usually, we use a split proportional to 50%/25%/25% for the three sets)

n_train = 20
n_vali = 10
n_test = 10

x_train, y_train = x[:n_train], y[:n_train]
x_vali, y_vali = x[n_train:n_train + n_vali], y[n_train:n_train + n_vali]
x_test, y_test = x[n_train + n_vali:], y[n_train + n_vali:]

plt.figure(figsize=(16,9))
plt.scatter(x_train, y_train, label='train')
plt.scatter(x_vali, y_vali, label='validation')
plt.scatter(x_test, y_test, label='test')

plt.plot(np.linspace(0,5, 100), fun(np.linspace(0,5,100)))
plt.grid()
plt.legend()
plt.xlabel('Input')
plt.ylabel('Target')
plt.savefig("ml/plot/bv_fig4.png", dpi=150)
# A plot of our function (fun(x)) and our dataset splitted


# To make the model better I can try to change the features
# Here I tranform each input x in a vector [1, x, x², x³] to see if it's better like this or not
# (It's like basis function)
# (h(x,w) is a linear model by being linear in w)

degree = 3
poly = PolynomialFeatures(degree=degree)
x_train_tr = poly.fit_transform(x_train)
print(x_train_tr)


# Now I do this with degree from 0 to 10, for each of them I train a model and
# for each I compute the mean sqare error
# Then I check with model is the best

models = [] # models are stored here
models_mse = [] # mean square errors are stored here

for degree in range(0, 10):
  # fit a linear model for each of the selected polynomial orders
  lin_model = linear_model.LinearRegression()
  poly = PolynomialFeatures(degree=degree)
  x_train_tr = poly.fit_transform(x_train)
  lin_model.fit(x_train_tr, y_train)

  # measure model performance
  y_train_pred = lin_model.predict(x_train_tr)
  mse = mean_squared_error(y_train, y_train_pred)

  # store
  models.append(lin_model)
  models_mse.append(mse)

# Now I plot the results to see how mse vary with the degree
plt.figure(figsize=(16,9))
plt.plot(models_mse)
plt.title('Only Training')
plt.grid()
plt.xlabel('Order')
plt.ylabel('MSE')
plt.savefig("ml/plot/bv_fig5.png", dpi=150)


# Here with the trainig dataset we did both trainnig and prediction
# Now we validation dataset and the test dataset we do only prediction
# and then we plot all the result togethere

models_mse_test = []
for degree in range(0, 10):
  # load the fitted model
  lin_model = models[degree]

  # compute polynomial features for the test
  poly = PolynomialFeatures(degree=degree)
  x_test_tr = poly.fit_transform(x_test)

  # measure model performance on TEST
  y_test_pred = lin_model.predict(x_test_tr)
  mse = mean_squared_error(y_test, y_test_pred)
  models_mse_test.append(mse)

plt.figure(figsize=(16,9))
plt.plot(models_mse, label='Training MSE')
plt.plot(models_mse_test, label='Test MSE')
plt.title('Training and test error')
plt.grid()
plt.xlabel('Order')
plt.ylabel('MSE')
plt.legend()


models_mse_vali = []
for degree in range(0, 10):
  # load the fitted model
  lin_model = models[degree]

  # compute polynomial features for the test
  poly = PolynomialFeatures(degree=degree)
  x_vali_tr = poly.fit_transform(x_vali)
  y_vali_pred = lin_model.predict(x_vali_tr)

  # measure model performance on TEST
  mse = mean_squared_error(y_vali, y_vali_pred)
  models_mse_vali.append(mse)

plt.figure(figsize=(16,9))
plt.plot(models_mse, label='Training MSE')
plt.plot(models_mse_test, label='Test MSE')
plt.plot(models_mse_vali, label='Validation MSE')

best_model = np.argmin(models_mse_vali)
plt.axvline(best_model)
plt.title('Training and test error')
plt.grid()
plt.xlabel('Order')
plt.ylabel('MSE')
plt.legend()


k_fold = 4
n_cross = n_train + n_vali
x_cross, y_cross = x[:n_cross], y[:n_cross]
cross_indices = list(range(len(x_cross)))
MSE_cross = np.zeros(10)
for degree in range(10):
  for k in range(k_fold):
    start_vali = int(np.round(n_cross * k / k_fold)) # start of the validation fold
    end_vali = int(np.round(n_cross * (k+1) / k_fold)) # end of the validation fold

    vali_indices = list(range(int(start_vali), int(end_vali)))
    train_indices = [i for i in cross_indices if i not in vali_indices] # train on all the other indices

    x_fold_train = x_cross[train_indices]
    y_fold_train = y_cross[train_indices]

    x_fold_vali = x_cross[vali_indices]
    y_fold_vali = y_cross[vali_indices]

    # train a model with polynomial feature
    lin_model = linear_model.LinearRegression()
    poly = PolynomialFeatures(degree=degree)
    x_train_tr = poly.fit_transform(x_fold_train)
    x_vali_tr = poly.fit_transform(x_fold_vali)
    lin_model.fit(x_train_tr, y_fold_train)

    # measure cross_validation error
    y_vali_pred = lin_model.predict(x_vali_tr)
    MSE = mean_squared_error(y_fold_vali, y_vali_pred)
    MSE_cross[degree] += MSE
  MSE_cross[degree] /= k_fold


plt.figure(figsize=(16,9))
plt.plot(models_mse, label='Training MSE')
plt.plot(models_mse_test, label='Test MSE')
plt.plot(models_mse_vali, label='Validation MSE')
plt.plot(MSE_cross, label='Cross-Validation MSE')

best_model = np.argmin(MSE_cross)
plt.axvline(best_model)
plt.title('Training and test error')
plt.grid()
plt.xlabel('Order')
plt.ylabel('MSE')
plt.legend()
plt.savefig("ml/plot/bv_fig6.png", dpi=150)


# From the image we can see that the order three is the best model according to cross validation


# Training MSE (blue) goes down almost every time the order increases, from about 10 to about 0.7.
# A more complex model always fits the training data better, so training error can’t be used to choose a model.
# It would pick order 9, which is the worst one on the test set.

# Order 0–2 (left side): underfitting. Every error is high, and training,
# validation and test are all bad together. The model is too simple to follow the curve,
# so the problem is high bias.

# Order 3–7 (middle): good region. Errors drop sharply at order 3 and stay fairly low.
# Order 3 is the simplest model that captures the true shape, so it’s the best choice.
# The extra degrees don’t add much.

# Order 8–9 (right side): overfitting. Training error is at its lowest (about 0.7),
# but test error explodes (25–28). The gap between training and test is huge.
# With only 20 training points, a degree 8–9 polynomial bends to pass through the noise,
# so the problem is high variance.

# Validation (green) and cross-validation (red) both reach their minimum around order 3.
# They follow the shape of the test error without touching the test set, which is the whole
# point of model selection.

# Don’t choose a model by training error, because it always rewards complexity and leads to overfitting.
# Choose it by error on data the model didn’t train on (validation, or better, cross-validation).
# That error forms a U shape: too simple underfits (high bias), too complex overfits (high variance),
# and the bottom of the U, here order 3, is the best balance. The test set is used only at the end,
# to check the chosen model.



# RECAP:

# Bias: the systematic error of the model, the gap between the truth f(x) and what the model learns
# on average over training sets. It comes from assumptions that are too restrictive
# (for example, a line for a curve).
# Variance: how much the learned model changes from one training set to another.
# It comes from a model that’s too sensitive to the specific data it saw.
# Underfitting: the model is too simple and misses the real pattern. Caused by high bias.
# Overfitting: the model is too complex and memorizes the noise. Caused by high variance.
# Training error: the error on the data used for training. It’s too optimistic.
# Validation error: the error on a separate set, used to choose the model.
# Cross-validation error: the validation error averaged over k splits.
# It’s a more reliable estimate of the error on new data.

# Training error   → always goes DOWN with complexity → don't use it to choose
# Validation error → U shape: down, then up          → its minimum = best model
# CV error         → same U, but more reliable        → preferred way to choose
# Test error       → measured only ONCE, at the end   → final performance