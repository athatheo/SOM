import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def load_data(choice):
    if choice == 1:
        dataset = load_iris()
        data = pd.DataFrame(dataset['data'], columns=['Petal length', 'Petal Width', "Sepal Length", "Sepal Width"])
        classes = pd.DataFrame(dataset['target'])
        return data, classes


def create_2d_map(side_len, input_dimensions):
    # Make a square map
    map = np.random.uniform(size=(side_len, side_len, input_dimensions))
    return map


def fit(epochs, map, data):
    prev_MAP = np.zeros((len(map),len(map[0]),4))
    convergence = [1]
    for epoch in range(epochs):
        print(epoch)
        # Randomize the order of the data
        data = data.sample(frac=1).reset_index(drop=True)
        learning_rate = update_learning_rate(epoch, epochs)
        radius = update_radius(epoch, epochs)
        neighborhood = update_neighborhood(learning_rate)

        for index, row in data.iterrows():
            map = update(row, map, learning_rate, radius, neighborhood)

        J = np.linalg.norm(map - prev_MAP)

        prev_MAP = np.copy(map)

        if J < min(convergence):
            print('Lower error found: %s' % str(J) + ' at epoch: %s' % str(epoch))
        convergence.append(J)

    return map, convergence, J


def find_minimum(vector, map):
    min = float('inf')
    x, y = 0, 0
    for i in range(len(map)):
        for j in range(len(map[0])):
            distance = euclid_distance(vector, map[i][j])
            if distance < min:
                min = distance
                x, y = i, j
    return x, y


def plot():
    plt.show()


def update(vector, map, learning_rate, radius, neighborhood):
    x, y = find_minimum(vector, map)
    bmu = map[x][y]
    for i in range(len(map)):
        for j in range(len(map[0])):
            if euclid_distance(np.asarray([i,j]), np.asarray([x,y])) < radius:
                map[i][j] = map[i][j] + learning_rate * neighborhood * (vector - map[i][j])
    return map


def update_radius(epoch, epochs):
    radius = 5
    return radius * math.exp(-epoch / epochs)


def update_neighborhood(learning_rate):
    neighborhood = 1
    # Bubble neighborhood function
    #neighborhood = learning_rate
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


# Get the data
# choice = int(input("Choose dataset: \n - For Iris dataset, type 1 \n"))
df, classes = load_data(1)  ##### REMEMBER TO CHANGE THIS TO CHOICE
classes = classes.to_numpy()
# Initialize the model
map = create_2d_map(6, len(df.columns))
# Train it
map, convergence, J = fit(epochs=200, map=map, data=df)
# Print results
'''
plt.plot(convergence)
plt.ylabel('error')
plt.xlabel('epoch')
plt.grid(True)
plt.yscale('log')
#plt.show()
print('Final error: ' + str(J))
'''
result_map = np.zeros([len(map), len(map[0]), 3], dtype=np.float32)
plt.clf()
i = 0
for index, row in df.iterrows():
    BMU = find_minimum(row, map)
    x = BMU[0]
    y = BMU[1]
    if classes[i] == 0:
        if result_map[x][y][0] <= 0.5:
            result_map[x][y] += np.asarray([0.5, 0, 0])
    elif classes[i] == 1:
        if result_map[x][y][1] <= 0.5:
            result_map[x][y] += np.asarray([0, 0.5, 0])
    elif classes[i] == 2:
        if result_map[x][y][2] <= 0.5:
            result_map[x][y] += np.asarray([0, 0, 0.5])
    i += 1
print(result_map)

print("Red = Iris-Setosa")
print("Blue = Iris-Virginica")
print("Green = Iris-Versicolor")

#print(map)
plt.imshow(result_map, interpolation='nearest')
plt.show()
