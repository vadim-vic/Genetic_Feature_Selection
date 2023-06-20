# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
# Datasets and models
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# Supress classification warnings to show genetic statistics
import warnings
# Plot the feature map in demo
import matplotlib.pyplot as plt

# Load the Digits data
digits = datasets.load_digits()
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

# Show the data structure
matrix = digits.images[-1]
vector = digits.data[-1]
print(matrix, vector)

# Each pixel of the picture acts as a feature in the logistic regression model
vector = np.random.randint(0, 2, size=64)
matrix = np.reshape(vector, (8, 8))  # This reshape will be used for visualization
plt.figure(1, figsize=(3, 3))
plt.imshow(matrix, cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
