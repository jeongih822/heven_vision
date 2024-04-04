import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    # step function that return 0 or 1
    if x < thres:
        return 0
    else:
        return 1

def gen_training_data(data_point):
    x1 = np.random.random(data_point)
    x2 = np.random.random(data_point)
    y = ((x1 + x2) > 1).astype(int) # 1 if (x1 + x2) > 1, 0 if (x1 + x2) <= 1
    training_set = [((x1[i], x2[i]), y[i]) for i in range(len(x1))]

    return training_set

thres = 0.5
w = np.array([0.3, 0.9]) # Initial weights
lr = 0.1 # Learning rate
data_point = 100
epoch = 10
training_set = gen_training_data(data_point)

plt.figure(0)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
ax = plt.gca()
ax.set_aspect('equal', adjustable = 'box')

# plot training set
for x, y in training_set:
    if y == 1:
        plt.plot(x[0], x[1], 'bo')
    else:
        plt.plot(x[0], x[1], 'go')
plt.show()

xx = np.linspace(0, 1, 50)
# Iterate over the epochs
for i in range(epoch):
    cnt = 0
    # Iterate over the training set
    for x, y in training_set:
        # Calculate the weighted sum
        u = sum(x * w)
        # Compute the error
        error = y - step_function(u)
        # Update the weights
        for index, value in enumerate(x):
            w[index] = w[index] + lr * error * value
        # Calculate the boundary
        yy = -w[1] / w[0] * xx + thres / w[0]
        cnt = cnt + 1

for xs, ys in training_set[0:100]:
    plt.ylim(-0.1, 1.1)
    plt.xlim(-0.1, 1.1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable = 'box')

    if ys == 1:
        plt.plot(xs[0], xs[1], 'bo')
    else:
        plt.plot(xs[0], xs[1], 'go')
plt.plot(xx, yy)
plt.show()