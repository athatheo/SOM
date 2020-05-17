import itertools
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer


def load_data(choice):
    if choice == 1:
        dataset = load_iris()
        data = pd.DataFrame(dataset['data'], columns=['Petal length', 'Petal Width', "Sepal Length", "Sepal Width"]).to_numpy()
        classes = np.squeeze(pd.DataFrame(dataset['target']).to_numpy())
    elif choice == 2:
        data, classes = load_breast_cancer(True)

    return data, classes


def create_2d_map(side_len, input_dimensions):
    # Make a square map
    map = np.random.uniform(size=(side_len, side_len, input_dimensions))
    return map


def fit(epochs, map, data):
    prev = np.zeros((len(map), len(map[0]), len(map[0][0])))
    temp_data = np.copy(data)
    for epoch in range(epochs):
        # Randomize the order of the data
        np.random.shuffle(temp_data)

        # Receive new parameters
        learning_rate = update_learning_rate(epoch, epochs)
        radius = update_radius(epoch, epochs, len(map))
        neighborhood = update_neighborhood(learning_rate)

        # Update the map for every input vector
        for row in temp_data:
            map = update(row, map, learning_rate, radius, neighborhood)

        # Metric to keep track of the convergence
        diff = euclid_distance(map, prev)
        prev = np.copy(map)

        print("Epoch:", epoch+1, "/", epochs, "Distance since previous map: ", diff)
    return map


def find_minimum(vector, map):
    min = float('inf')
    bmu_row, bmu_col = 0, 0
    for row, col in itertools.product(range(len(map)), range(len(map[0]))):
        distance = euclid_distance(vector, map[row][col])
        if distance < min:
            min = distance
            bmu_row, bmu_col = row, col
    return bmu_row, bmu_col


def update(vector, map, learning_rate, radius, neighborhood):
    bmu_row, bmu_col = find_minimum(vector, map)
    for row, col in itertools.product(range(len(map)), range(len(map[0]))):
        if euclid_distance(np.asarray([row, col]), np.asarray([bmu_row, bmu_col])) < radius:
            map[row][col] = map[row][col] + learning_rate * neighborhood * (vector - map[row][col])
    return map


def update_radius(epoch, epochs, length_map):
    radius = length_map/2
    return radius * math.exp(-epoch / epochs)


def update_neighborhood(learning_rate):
    neighborhood = 1
    # Bubble neighborhood function
    # neighborhood = learning_rate
    return neighborhood


def update_learning_rate(epoch, epochs):
    learning_rate = 0.25
    factor = 1 - (epoch / epochs)
    # factor = 1/iteration
    # factor = math.exp(iteration/epochs)
    learning_rate = learning_rate * factor
    return learning_rate


def euclid_distance(input_vector, unit):
    return np.linalg.norm(input_vector - unit)


def plot(df, map, classes):
    colored_map = np.zeros([len(map), len(map[0]), 3])
    for i, row in zip(range(len(df)), df):
        x, y = find_minimum(row, map)
        if classes[i] == 0:
            if colored_map[x][y][0] <=0.5:
                colored_map[x][y] += np.asarray([0.5, 0, 0])
        elif classes[i] == 1:
            if colored_map[x][y][1] <=0.5:
                colored_map[x][y] += np.asarray([0, 0.5, 0])
        elif classes[i] == 2:
            if colored_map[x][y][2] <=0.5:
                colored_map[x][y] += np.asarray([0, 0, 0.5])
    plt.imshow(colored_map)
    plt.show()


# Get the data
choice = int(input("Choose dataset: \n - For Iris dataset, type 1 \n - For Breast Cancer dataset, type 2\n"))
df, classes = load_data(choice)
# Initialize the model
map = create_2d_map(10, len(df[0]))
# Train it
map = fit(epochs=500, map=map, data=df)

# Print results
plot(df, map, classes)


