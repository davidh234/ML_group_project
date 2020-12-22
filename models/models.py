import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor
from scipy.stats import norm

# read in the data set

df = pandas.read_csv("subset-kerry.csv", header=None, comment='#')

# parse each of the features into a single column list
X1 = df.iloc[:, 0]  # Average_House_price_normalized
X2 = df.iloc[:, 1]  # Average_hotel_cost_normalized
X3 = df.iloc[:, 2]  # beds
X4 = df.iloc[:, 3]  # accommodates
X5 = df.iloc[:, 4]  # host_total_listings_count
X6 = df.iloc[:, 5]  # host_identity_verified
X7 = df.iloc[:, 6]  # reviews_per_month
X8 = df.iloc[:, 7]  # host_is_superhost
X9 = df.iloc[:, 8]  # review_scores_rating_normalized
y = df.iloc[:, 9]   # price
X = np.column_stack((X1, X2, X3, X4, X5, X6, X7, X8, X9))

# baseline model used is the DummyRegressor model, with a strategy of 'mean'
# This model takes the mean price of the given data set and predicts it each time
dummy_model = DummyRegressor(strategy='mean')
dummy_model.fit(X, y)
dummy_y_accuracy = dummy_model.predict(X)
print("Dummy MSE accuracy:", mean_squared_error(y, dummy_y_accuracy))
print("Dummy R2 accuracy:", r2_score(y, dummy_y_accuracy))


# helper function for plotting price data as a normal distribution to observe the distribution of prices in the data set
def plot_price_data(prices):
    prices = np.array(prices)
    prices.sort()
    std = np.std(prices)
    mean = np.mean(prices)
    pdf = norm.pdf(prices, mean, std)
    plt.plot(prices, pdf)
    plt.show()


# given a feature set X and corresponding target variable Y as well as a list of C values for L1 regularisation,
# train a LASSO regression model on the training data, and measure its prediction capabilities on the test data
def lasso_model(X, y, C_values):
    estimate_list = []
    average_lowest_mse = 0
    # split the data into train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    lowest_mse = 999999
    # for experimentation with Polynomial Features
    # Xtrain_poly = PolynomialFeatures(3).fit_transform(X_train)
    # Xtest_poly = PolynomialFeatures(3).fit_transform(X_test)

    for c in C_values:
        # configure the model and train it on the training data
        model = Lasso(alpha=(1 / (2 * c)))
        model.fit(X_train, y_train)
        # using the test set, test the models prediction capability on unseen data
        ypred = model.predict(X_test)
        # calculate the MSE as a measure of accuracy to compare against other models
        mse = mean_squared_error(y_test, ypred)
        r_squared = r2_score(y_test, ypred)
        if mse < lowest_mse:
            lowest_mse = mse
            print("R2:", r_squared)

        estimate_list.append(mse)
    # if average_lowest_mse != 0:
    #     average_lowest_mse = (average_lowest_mse + lowest_mse) / 2
    # else:
    #     average_lowest_mse = lowest_mse
    print("LASSO best accuracy:", lowest_mse)

    # add error bar and title, label axes
    plt.errorbar(C_values, estimate_list)
    plt.xlabel("C value")
    plt.ylabel("Mean Square error")
    plt.title("Comparison - MSE and C values for Lasso model")
    plt.show()


# given a feature set X and corresponding target variable Y as well as a list of C values for L2 regularisation,
# train a Ridge regression model on the training data, and measure its prediction capabilities on the test data
def ridge_model(X, y, C_values):
    estimate_list = []
    average_lowest_mse = 0
    # split the data into train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    lowest_mse = 999999

    for c in C_values:
        # configure the model and train it on the training data
        model = Ridge(alpha=(1 / (2 * c)))
        model.fit(X_train, y_train)
        # using the test set, test the models prediction capability on unseen data
        ypred = model.predict(X_test)
        # calculate the MSE as a measure of accuracy to compare against other models
        mse = mean_squared_error(y_test, ypred)
        r_squared = r2_score(y_test, ypred)
        if mse < lowest_mse:
            lowest_mse = mse
            print("R2: ", r_squared)
        estimate_list.append(mse)
        # if average_lowest_mse != 0:
        #     average_lowest_mse = (average_lowest_mse + lowest_mse) / 2
        # else:
        #     average_lowest_mse = lowest_mse
    print("Ridge best accuracy:", lowest_mse)

    # add error bar and title, label axes
    plt.errorbar(C_values, estimate_list)
    plt.xlabel("C value")
    plt.ylabel("Mean Square error")
    plt.title("Comparison - MSE and C values for Ridge model")
    plt.show()


# given a feature set X and corresponding target variable Y, fit a KNN model to the data and test its predictions
# with a given number of neighbours = k. Record the best performing model and plot the performance
def knn_model(X, y, num_neighbours_list):
    estimate_list = []
    lowest_mse = 9999
    lowest_mse_neighbours = 9999
    average_lowest_mse = 0

    # split the data into train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    for num_neighbours in num_neighbours_list:
        # configure the model and train it on the training data
        model = KNeighborsRegressor(n_neighbors=num_neighbours, weights="distance").fit(X_train, y_train)
        # using the test set, test the models prediction capability on unseen data
        ypreds = model.predict(X_test)
        # calculate the MSE as a measure of accuracy to compare against other models
        mse = mean_squared_error(y_test, ypreds)
        r_squared = r2_score(y_test, ypreds)
        print("R2:", r_squared)
        estimate_list.append(mse)
        if mse < lowest_mse:
            lowest_mse = mse
            lowest_mse_neighbours = num_neighbours
        # if average_lowest_mse != 0:
        #     average_lowest_mse = (average_lowest_mse + lowest_mse) / 2
        # else:
        #     average_lowest_mse = lowest_mse
    print("best KNN - nn: ", lowest_mse_neighbours, ", mse:", lowest_mse)

    # add error bar and title, label axes
    plt.errorbar(num_neighbours_list, estimate_list)
    plt.xlabel("# neighbours")
    plt.ylabel("Mean Square error")
    plt.title("Comparison - MSE and Number of Neighbours for KNN model")
    plt.show()


# driver code

# set the different number of neighbours the KNN model will try
num_neighbours_list = [1, 2, 5, 10, 25, 50, 100]

# the list of C values used for L1 regularisation in the LASSO regression model
lasso_C_values = [0.1, 1, 10, 25, 50]

# the list of C values used for L2 regularisation in the Ridge regression model
ridge_C_values = [0.00001, 0.0001, 0.001]  # best is 0.0001 with .1 split

# invoking each of the models
knn_model(X, y, num_neighbours_list)
lasso_model(X, y, lasso_C_values)
ridge_model(X, y, ridge_C_values)
