import numpy as np


class ActivationFunctions:

    def multip(self):
        return 1 * self.index


class Sigmoid(ActivationFunctions):
    index = 1

    def __call__(self, x):
        return 1 / (np.exp(-x) + 1)

    def calculate(self, x):
        return self.__call__(x)

    def inverse(self, x):
        return self.__call__(x)(1 - self.__call__(x))


class Relu(ActivationFunctions):
    index = 0

    def __init__(self, lambd=0):
        self.lambd = lambd

    def __call__(self, x):
        return np.maximum(np.zeros_like(x) + self.lambd * x, x)

    def calculate(self, x):
        return self.__call__(x)

    def inverse(self, x):
        if x >= 0:
            return 1
        return self.lambd


class Tanh(ActivationFunctions):
    index = 0

    def __call__(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def calculate(self, x):
        return self.__call__(x)

    def inverse(self, x):
        return 1 - self.__call__(x) ** 2


class Softmax():
    index = 0

    def __call__(self, x):
        pass

    def calculate(self, x):
        return self.__call__(x)


# y shape 1*a
# x shape n*m
# wl shape nl*nl-1
def compute_cost(y_exp, y):
    m = y.shape[1]
    return -1 / m * (np.dot(y, np.log(y_exp).T) + np.dot((1 - y), np.log(1 - y_exp).T))


def predict():
    return 0


def initialise_parameters(layers):
    parameters = {}
    for i in range(len(layers) - 1):
        parameters["w" + str(i + 1)] = np.random.randn(layers[i + 1], layers[i])
        parameters["b" + str(i + 1)] = np.zeros((layers[i + 1], 1))
    return parameters


def forward_propagate(a_prev, w, b, activation=Relu()):
    z = np.dot(w, a_prev) + b
    a = activation.calculate(z)
    return a, z


def backward_propagate(da, a, z, activation=Relu()):
    dz = activation.inverse(a)
    dw = None
    db = None
    da_prev = None
    return da_prev, dw, db


# x,y3,[1,2,3],[a1,a2,a3],alp,ilter
def deep_nn(x, y, layers, activations, no_of_ilter=500, learning_rate=0.001, freq=0):
    n, m = x.shape
    costs = []
    cost = 0
    cache = {}
    num_of_layers = len(layers)
    # initialise parameters
    parameters = initialise_parameters(layers)

    for i in range(no_of_ilter):
        # forward propagate
        A = x
        for i in range(1, num_of_layers):
            A_prev = A
            A, z = forward_propagate(A_prev, parameters['w' + str(i)], activations[i])
            cache["a" + str(i)], cache["z" + str(i)] = A, z
        # compute cost
        cost = compute_cost(A, y)
        print(f"cost is {cost}", end='r')
        if freq and i % freq == 0:
            costs.append(cost)
        # backward propagate
        da = None
        grads = {}
        for i in range(1, num_of_layers):
            l = num_of_layers - i
            da_prev, dw, db = backward_propagate(da, cache["a" + str(l)], cache["z" + str(l)], activations[l - 1])
            da = da_prev
            grads["dw" + str(l)] = dw
            grads["db" + str(l)] = db

        # update variables
        for i in range(num_of_layers):
            parameters["w" + str(i + 1)] = parameters["w" + str(i + 1)] + learning_rate * grads["w" + str(i + 1)]
            parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] + learning_rate * grads["b" + str(i + 1)]

    print(f"cost is {cost}")

    return parameters, costs

# print(deep_nn(np.array([[1, 2, 3], [1, 2, 3]]), 1, [2, 3, 4, 5], 1))
# act = Sigmoid()
# print(act.multip())
