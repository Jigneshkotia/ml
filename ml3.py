import pandas as pd
import numpy as np

# Load the Iris dataset
iris_df = pd.read_csv('Iris.csv')

# Display the first few rows of the dataset
print("First few rows of the Iris dataset:")
print(iris_df.head())

# Selecting features and target variable
X = iris_df[['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]  # Features
y = iris_df['SepalLengthCm']  # Target variable

# Add a column of ones to the features for the intercept term
X['intercept'] = 1

# Convert features and target variable to NumPy arrays
X = X.values
y = y.values.reshape(-1, 1)  # Reshape y to ensure it's a 2D array

# Calculate the coefficients using the normal equation: theta = (X^T * X)^-1 * X^T * y
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Make predictions
y_pred = X.dot(theta)

# Calculate mean squared error
mse = np.mean((y - y_pred)**2)

# Calculate R-squared score
ssr = np.sum((y_pred - np.mean(y))**2)
sst = np.sum((y - np.mean(y))**2)
r_squared = ssr / sst

print("\nMean Squared Error:", mse)
print("R-squared Score:", r_squared)
