import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data = pd.read_csv('Housing.csv')
data.head()
data.tail(5)
print("Rows and Columns of the dataset :- ",data.shape)
data.info()
data.columns
data.describe(include ='all')
data.isnull().sum()
categorical_col =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
data[categorical_col]
def binary_map(x):
    """
    Function to map 'yes' and 'no' values to 1 and 0, respectively.

    Parameters:
    x (pandas Series): Input Series containing 'yes' and 'no' values.

    Returns:
    pandas Series: Mapped Series with 'yes' mapped to 1 and 'no' mapped to 0.
    """
    return x.map({'yes': 1, 'no': 0})
data[categorical_col] = data[categorical_col].apply(binary_map)
data[categorical_col]
data.head()
dummy_col = pd.get_dummies(data['furnishingstatus'])
dummy_col.head()
dummy_col = pd.get_dummies(data['furnishingstatus'], drop_first=True)
dummy_col.head()
data = pd.concat([data, dummy_col], axis=1)
data.head()
data.drop(['furnishingstatus'], axis=1, inplace=True)
data.head()
data.columns
np.random.seed(0)
df_train, df_test = train_test_split(data, train_size=0.7, test_size=0.3, random_state=100)
df_train.head()
df_train.shape
df_test.head()
df_test.shape
scaler = MinMaxScaler()
col_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
df_train[col_to_scale] = scaler.fit_transform(df_train[col_to_scale])
y_train = df_train.pop('price')
x_train = df_train
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)
coefficients = linear_regression.coef_
print(coefficients)
score = linear_regression.score(x_train, y_train)
print(score)
col_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
df_test[col_to_scale] = scaler.fit_transform(df_test[col_to_scale])
y_test = df_test.pop('price')
x_test = df_test
prediction = linear_regression.predict(x_test)
r2 = r2_score(y_test, prediction)
y_test.shape
y_test_matrix = y_test.values.reshape(-1, 1)
data_frame = pd.DataFrame({'actual': y_test_matrix.flatten(), 'predicted': prediction.flatten()})
data_frame.head(10)
fig = plt.figure()
plt.scatter(y_test, prediction)
plt.title('Actual vs Prediction')
plt.xlabel('Actual', fontsize=15)
plt.ylabel('Predicted', fontsize=15)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, prediction)
print("Mean Squared Error:", mse)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(x_train, y_train)
knn_y_pred = knn_model.predict(x_test)
knn_mse = mean_squared_error(y_test, knn_y_pred)
knn_r2 = r2_score(y_test, knn_y_pred)
print(" Mean Squared Error:", knn_mse)
print(" R-squared:", knn_r2)
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(x_train, y_train)
dt_y_pred = dt_model.predict(x_test)
dt_mse = mean_squared_error(y_test, dt_y_pred)
dt_r2 = r2_score(y_test, dt_y_pred)
print(" Mean Squared Error:", dt_mse)
print(" R-squared:", dt_r2)
import matplotlib.pyplot as plt
algorithms = ['Linear Regression', 'KNN', 'Decision Tree']
r2_scores = [r2, knn_r2, dt_r2]
plt.figure(figsize=(10, 6))
plt.bar(algorithms, r2_scores, color=['blue', 'green', 'red'])
plt.title('R-squared Comparison')
plt.xlabel('Algorithm')
plt.ylabel('R-squared')
plt.ylim(0, 1)
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(y_test, knn_y_pred, color='blue')
plt.title('Actual vs. Predicted Median House Value')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(y_test, dt_y_pred, color='blue')
plt.title('Actual vs. Predicted Median House Value')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.show()