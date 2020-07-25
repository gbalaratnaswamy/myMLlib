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
    return -1 / (2 * m) * (np.dot(y, np.log(y_exp).T) + np.dot((1 - y), np.log(1 - y_exp).T))


def predict():
    return 0


# x,y3,[1,2,3],[a1,a2,a3],alp,ilter
def deep_nn(x, y, layers, activations, no_of_ilter=500, learning_rate=0, freq=0):
    n, m = x.shape
    parameters = {}
    costs = []
    cost = 0
    # initialise parameters
    for i in range(len(layers) - 1):
        parameters["w" + str(i + 1)] = np.random.randn(layers[i + 1], layers[i])
        parameters["b" + str(i + 1)] = np.zeros((layers[i + 1], 1))

    for i in range(no_of_ilter):
        # forward propagate
        y_exp = 0

        # compute cost
        cost = compute_cost(y_exp, y)
        print(f"cost is {cost}", end='r')
        if freq and i % freq == 0:
            costs.append(cost)
        # backward propagate

        # update variables

    print(f"cost is {cost}")

    return parameters, costs

# print(deep_nn(np.array([[1, 2, 3], [1, 2, 3]]), 1, [2, 3, 4, 5], 1))
# act = Sigmoid()
# print(act.multip())
