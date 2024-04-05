import pandas as pd
import numpy as np

# Load the dataset
housing_df = pd.read_csv('Housing.csv')



# Separate features and target variable
X = housing_df.drop(columns=['price'])  # Features
y = housing_df['price']  # Target variable

# Perform one-hot encoding on categorical columns
X = pd.get_dummies(X, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'])

# Convert features and target variable to NumPy arrays
X = X.values
y = y.values.reshape(-1, 1)  # Reshape y to ensure it's a 2D array

# Add a column of ones to the features for the intercept term
X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

# Perform linear regression using the normal equation: theta = (X^T * X)^-1 * X^T * y
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
