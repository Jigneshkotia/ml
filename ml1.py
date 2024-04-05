import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Iris dataset
iris_df = pd.read_csv('Iris.csv')

# Display the first few rows of the dataset
print("First few rows of the Iris dataset:")
print(iris_df.head())



# Encode categorical variable 'species' using one-hot encoding
iris_encoded = pd.get_dummies(iris_df, columns=['Species'])

# Display the first few rows of the encoded dataset
print("\nEncoded dataset:")
print(iris_encoded.head())

# Scale features using Min-Max scaling
def min_max_scaling(data):
    return (data - data.min()) / (data.max() - data.min())

# Identify numerical columns for scaling
numerical_columns = iris_encoded.columns[:-3]  # Exclude encoded species columns
iris_scaled_df = iris_encoded.copy()  # Create a copy to keep original data intact

# Apply Min-Max scaling to numerical columns
iris_scaled_df[numerical_columns] = iris_scaled_df[numerical_columns].apply(min_max_scaling)

# Display the first few rows of the scaled dataset
print("\nScaled dataset:")
print(iris_scaled_df.head())

# # Plot histogram of features
iris_scaled_df[numerical_columns].hist(bins=20, figsize=(10, 6), color='skyblue', edgecolor='black')
plt.suptitle('Histogram of Scaled Features', fontsize=16)
plt.show()


