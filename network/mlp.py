import numpy as np
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        # self.w1_2_3_4 = np.random.random((self.input_size, self.hidden_size))
        # self.w5_6 = np.random.random((self.hidden_size, self.output_size))
        self.w1_2_3_4 = [[1, 10], [1, 10]]
        self.w5_6 = [[-40], [40]]

    # Sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        # Propagate inputs through the network
        self.z1_2 = np.dot(X, self.w1_2_3_4) # 1 X 2
        self.h = self.sigmoid(self.z1_2) # 1 X 2
        self.z3 = np.dot(self.h, self.w5_6) # 1 X 1
        self.o = self.sigmoid(self.z3) # 1 X 1
        return self.o
    
    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, X, y, y_pred, learning_rate):
        # Chain Rule
        dc_do1 = -2 * (y - y_pred)
        do1_dz3 = y_pred * (1 - y_pred) # derivative of sigmoid
        dz3_dw5_6 = self.h
        # Compute the gradient of weights using error gradient and activation function gradient
        dc_dw5_6 = dc_do1 * do1_dz3 * dz3_dw5_6
        # Update weights
        self.w5_6 = self.w5_6 - learning_rate * dc_dw5_6.T
        # Backpropagate to compute gradients of hidden layer weights
        dc_dw1_2_3_4 = dc_do1 * do1_dz3 * np.dot(self.w5_6 * (self.h * (1 - self.h)).T, X)
        # Update weights
        self.w1_2_3_4 = self.w1_2_3_4 - learning_rate * dc_dw1_2_3_4.T

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(len(x_train)):
                y_pred = self.forward([x_train[i]])
                loss = self.mse_loss(y_train[i], y_pred)
                self.backward([x_train[i]], [y_train[i]], y_pred, learning_rate)
            if np.mod(epoch, 100) == 0:
                print('epoch = ', epoch, 'loss = ',loss)

x_train = np.random.randint(0, 2, (100, 2))
y_train = (x_train[:,0] != x_train[:,1].astype(int))

mlp = MLP(input_size = 2, hidden_size = 2, output_size = 1)
mlp.train(x_train, y_train, epochs = 1000, learning_rate = 0.1)

test_input = np.array([[0,0]])
predicted_output = mlp.forward(test_input)
print("Predicted Output : ", test_input, predicted_output)

test_input = np.array([[1,0]])
predicted_output = mlp.forward(test_input)
print("Predicted Output : ", test_input, predicted_output)

test_input = np.array([[0,1]])
predicted_output = mlp.forward(test_input)
print("Predicted Output : ", test_input, predicted_output)

test_input = np.array([[1,1]])
predicted_output = mlp.forward(test_input)
print("Predicted Output : ", test_input, predicted_output)