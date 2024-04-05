import numpy as np

# Loading Iris dataset
iris_data = np.genfromtxt('Iris.csv', delimiter=',', skip_header=1, usecols=(0, 1, 2, 3))

# Display the first few rows of the dataset
print("First few rows of the Iris dataset:")
print(iris_data[:5])

print("\n\n")



import matplotlib.pyplot as plt

petal_lengths = iris_data[:, 2]
plt.hist(petal_lengths, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')
plt.title('Histogram of Petal Lengths')
plt.show()


import pandas as pd

# Loading Iris dataset again for pandas
iris_df = pd.read_csv('Iris.csv')

print("First few rows of the Iris dataset:")
print(iris_df.head())
