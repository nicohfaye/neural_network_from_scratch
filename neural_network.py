import numpy as np
import matplotlib.pyplot as plt
from data_handler import get_mnist


images, lables = get_mnist()

# The layers in the network, initialized with random values
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))


learning_rate = 0.01
nr_correct = 0
epochs = 3

def sigmoid(h):
    return 1 / (1 + np.exp(-h))

for epoch in range(epochs):
    for img, l in zip(images, lables):
        img.shape += (1,)
        l.shape += (1,)
        # Forward propogation : input -> hidden layer
        h_pre = b_i_h + w_i_h @ img
        # Passed through activation function
        h = sigmoid(h_pre)
        # Forward propogation hidden layer -> output layer
        o_pre = b_h_o + w_h_o @ h
        o = sigmoid(o_pre)

        # Error calculation - Mean Squared Error (MSE)
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # Backpropogation output layer -> hidden layer (cost function derivative)
        delta_o = o - l
        w_h_o += -learning_rate * delta_o @ np.transpose(h)
        b_h_o += -learning_rate * delta_o
        # Backprop hidden layer -> input layer (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learning_rate * delta_h @ np.transpose(img)
        b_i_h += -learning_rate * delta_h

    # Show the accuracy for the current epoch
    print(f'Accuracy: {round((nr_correct / images.shape[0]) * 100, 2)}%')
    nr_correct = 0

# Show the results

while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"It is probably a {o.argmax()} :)")
    plt.show()


