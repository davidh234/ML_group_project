import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from scipy.stats import norm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors._dist_metrics import DistanceMetric

df = pandas.read_csv("group_project_data.csv", header=None, comment='#')

X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X3 = df.iloc[:, 2]
X4 = df.iloc[:, 3]
X5 = df.iloc[:, 4]
X6 = df.iloc[:, 5]
X7 = df.iloc[:, 6]
X8 = df.iloc[:, 7]
X9 = df.iloc[:, 8]
y = df.iloc[:, 9]
X = np.column_stack((X1, X2, X3, X4, X5, X6, X7, X8, X9))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# C = 1
# model = Lasso(alpha=(1 / (2 * C)))
# model.fit(X_train, y_train)
# y_accuracy = model.predict(X_test)
# print("LASSO accuracy:", mean_squared_error(y_test, y_accuracy))

# ridge_C = 1
# ridge_model = Ridge(alpha=(1 / (2 * ridge_C)))
# ridge_model.fit(X_train, y_train)
# ypred = ridge_model.predict(X_test)
# print("Ridge accuracy:", mean_squared_error(y_test, ypred))

dummy_model = DummyRegressor(strategy='mean')
dummy_model.fit(X_train, y_train)
dummy_y_accuracy = dummy_model.predict(X_test)
print("Dummy accuracy:", mean_squared_error(y_test, dummy_y_accuracy))


def plot_price_data(prices):
    prices = np.array(prices)
    prices.sort()
    std = np.std(prices)
    mean = np.mean(prices)
    pdf = norm.pdf(prices, mean, std)
    plt.plot(prices, pdf)
    plt.show()


# plot_price_data(y)

def lasso_calculate_prediction_error(X, y, C_values):
    estimate_list = []
    lowest_mse = 999999
    for c in C_values:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # for each iteration, set the model, train it on training data and test it on test data, then
        # compare the predictions to the actual target values
        model = Lasso(alpha=(1 / (2 * c)))
        model.fit(X_train, y_train)
        ypred = model.predict(X_test)
        mse = mean_squared_error(y_test, ypred)
        if mse < lowest_mse:
            lowest_mse = mse
        estimate_list.append(mse)
        # print(model.coef_)

    print("LASSO best accuracy:", lowest_mse)
    # add error bar and title, label axes
    plt.errorbar(C_values, estimate_list)
    plt.xlabel("C value")
    plt.ylabel("Mean Square error")
    plt.title("Comparison - MSE and C values for Lasso model")
    plt.show()


def ridge_calculate_prediction_error(X, y, C_values):
    estimate_list = []
    lowest_mse = 999999
    for c in C_values:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # for each iteration, set the model, train it on training data and test it on test data, then
        # compare the predictions to the actual target values
        model = Ridge(alpha=(1 / (2 * c)))
        model.fit(X_train, y_train)
        ypred = model.predict(X_test)
        mse = mean_squared_error(y_test, ypred)
        if mse < lowest_mse:
            lowest_mse = mse
        estimate_list.append(mse)
        # print(mse)

    print("Ridge best accuracy:", lowest_mse)
    # add error bar and title, label axes
    plt.errorbar(C_values, estimate_list)
    plt.xlabel("C value")
    plt.ylabel("Mean Square error")
    plt.title("Comparison - MSE and C values for Ridge model")
    plt.show()


def knn_model(num_neighbours_list):
    for num_neighbours in num_neighbours_list:
        model = KNeighborsRegressor(n_neighbors=num_neighbours, weights="distance").fit(X_train, y_train)
        ypreds = model.predict(X_test)
        mse = mean_squared_error(y_test, ypreds)
        print("nn: ", num_neighbours, ", mse:", mse)


# haversine_dist()
knn_model([1, 2, 5, 10, 25, 50, 100, 500])

lasso_calculate_prediction_error(X, y, [0.1, 1, 10, 25, 50])
#
ridge_calculate_prediction_error(X, y, [0.0001, 0.001, 0.1, 1, 5])
