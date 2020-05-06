import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def load_data(choice):
    if choice == 1:
        dataset = load_iris()
        data = pd.DataFrame(dataset['data'], columns=['Petal length', 'Petal Width', "Sepal Length", "Sepal Width"])
        return data


def create_2d_map(side_len, input_dimensions):
    # Make a square map
    map = np.random.uniform(size=(side_len, side_len, input_dimensions))
    return map


def fit(epochs, map, data):
    for epoch in range(epochs):
        # Randomize the order of the data
        data = data.sample(frac=1).reset_index(drop=True)
        for iteration, row in data.iterrows():
            vector = [row[0], row[1], row[2], row[3]]
            map = update(vector, map, iteration, epochs)
    return map


def find_minimum(vector, map):
    min = float('-inf')
    x, y = 0, 0
    for i in range(len(map)):
        for j in range(len(map[0])):
            if np.linalg.norm(vector - map[i][j]) < min:
                min = euclid_distance(vector, map[i][j])
                x, y = i, j
    return x, y


def plot():
    plt.show()


def update(vector, map, iteration, epochs):
    x, y = find_minimum(vector, map)
    bmu = map[x][y]
    learning_rate = update_learning_rate(iteration, epochs)
    radius = update_radius(iteration, epochs)
    neighborhood = update_neighborhood(learning_rate)
    for i in range(len(map)):
        for j in range(len(map[0])):
            if euclid_distance(map[i][j], bmu) < radius:
                map[i][j] = map[i][j] + learning_rate * neighborhood * (vector - map[i][j])
    return map


def update_radius(iteration, epochs):
    radius = 5
    return radius * math.exp(-iteration / epochs)


def update_neighborhood(learning_rate):
    # Bubble neighborhood function
    # neighborhood = learning_rate
    return 1


def update_learning_rate(iteration, epochs):
    learning_rate = 0.25
    factor = 1 - iteration / epochs
    # factor = 1/iteration
    # factor = math.exp(iteration/epochs)
    learning_rate = learning_rate * factor
    return learning_rate


def euclid_distance(input_vector, unit):
    return np.linalg.norm(input_vector - unit)


# Get the data
# choice = int(input("Choose dataset: \n - For Iris dataset, type 1 \n"))
df = load_data(1)  ##### REMEMBER TO CHANGE THIS TO CHOICE

# Initialize the model
map = create_2d_map(5, len(df.columns))

# Train it
map = fit(epochs=1, map=map, data=df)
# Print results
