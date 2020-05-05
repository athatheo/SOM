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
        for index, row in data.iterrows():
            vector = [row[0], row[1], row[2], row[3]]
            map = update(vector, map)
    return map


def find_minimum(vector, map):
    min = float('-nf')
    x, y = 0, 0
    for i in range(len(map)):
        for j in range(len(map[0])):
            if np.linalg.norm(vector - map[i][j]) < min:
                x, y = i, j
    return x, y



def plot():
    plt.show()


def update(vector, map):
    x, y = find_minimum(vector, map)


def neighborhoods(min):
    return map[min], map[2]


def distance(input_vector, unit):
    return input_vector - unit


# Get the data
# choice = int(input("Choose dataset: \n - For Iris dataset, type 1 \n"))
df = load_data(1)  ##### REMEMBER TO CHANGE THIS TO CHOICE

# Initialize the model
map = create_2d_map(5, len(df.columns))
print(map)
print(map[0])
# Train it
map = fit(epochs=1, map=map, data=df)

# Print results
