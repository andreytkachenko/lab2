import numpy as np
import pickle
from mlxtend.data import loadlocal_mnist
import math as m
from numba import jit

np.random.seed(1)                   # заставим numpy выдавать одинаковые набор случайных чисел для каждого запуска программы
np.set_printoptions(suppress=True)  # выводить числа в формате 0.123 а не 1.23e-1

# Загрузка датасета:

# В `X` находятся изображения для обучения, а в `y` значения соответственно
# `X.shape` == (60000, 784)   # изображения имеют размер 28x28 pix => 28*28=784
# `y.shape` == (60000,)       # каждое значение это число от 0 до 9 то что изображено на соответствующем изображении 
X, y = loadlocal_mnist(
        images_path="/home/andrey/datasets/mnist/train-images-idx3-ubyte", 
        labels_path="/home/andrey/datasets/mnist/train-labels-idx1-ubyte")

# В `Xt` находятся изображения для тестирования, а в `yt` значения соответственно
# `Xt.shape` == (10000, 784)   # изображения имеют размер 28x28 pix => 28*28=784
# `yt.shape` == (10000,)       # каждое значение это число от 0 до 9 то что изображено на соответствующем изображении 
Xt, yt = loadlocal_mnist(
        images_path="/home/andrey/datasets/mnist/t10k-images-idx3-ubyte", 
        labels_path="/home/andrey/datasets/mnist/t10k-labels-idx1-ubyte")

#@jit(nopython=True)
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1.0 / (1.0 + np.exp(-x))

#@jit(nopython=True)
def convert(y):
    y_d = np.zeros((len(y), 10))

    for idx, val in enumerate(y):
        y_d[idx, val] = 1.0

    return y_d

# Нормализуем значения изображений

X = X * (1 / 255)
Xt = Xt * (1 / 255)


# Параметры:

lr = 1       # значени на которое будет домножаться дельта на каждом шаге
batch = 60   # кол-во изображений использованное для обучения на каждом шаге
epochs = 100  # кол-во эпох. Если видно что прогресс есть, но нужно больше итераций 

# Объявляем веса:

def train(epochs, lr, batch_size, X, y, Xt, yt):
    W1 = np.random.uniform(-0.05, 0.05, (784, 16)) # инициализация случайными числами с равномерным распределением (uniform distribution) в диапазоне от -0.05 до 0.05
    W2 = np.random.uniform(-0.05, 0.05, (16, 10)) # тоже самое
    b1 = np.zeros((16,))
    b2 = np.zeros((10,))

    # Объявляем дельты

    dW1 = np.zeros((784, 16))
    dW2 = np.zeros((16, 10))

    # Обучение:

    batch_count = m.ceil(len(y) / batch)

    t = np.zeros((len(y), 10))
    np.put_along_axis(t, y.reshape((-1, 1)), 1.0, axis=1)
    
    #t = convert(y)

    for epoch in range(0, epochs): 
        # нужно сделать цикл на каждой итерации которого последовательно выбирать `batch` примеров из `x`
        # например индексы для 60 элементов в батче:
        #   итерация 1 - от 0 до 59
        #   итерация 2 - от 60 до 119
        #   итерация 3 - от 120 до 179 
        #   и т.д.

        if epoch < 10:
            lr = 1
        elif epoch < 20:
            lr = 0.5
        else:
            lr = 0.1

        for (bX, bt) in zip(np.split(X, batch_count), np.split(t, batch_count)):
            # forward 
            h1 = bX.dot(W1) + b1
            h2 = sigmoid(h1)
            h3 = h2.dot(W2) + b2
            o  = sigmoid(h3)

            # loss
            e = 2 * (o - bt)

            # backward
            z3 = e * sigmoid(o, True)
            z2 = np.dot(z3, W2.T)
            z1 = z2 * sigmoid(h2, True)
            # z4 = np.dot(z3, w1.t)

            # вычисление дельт для `w1`, `w2`, `b1` и `b2`
            dW1 = np.dot(bX.T, z1)
            dW2 = np.dot(h2.T, z3)
            db1 = np.mean(z1, axis=0)
            db2 = np.mean(z3, axis=0)
            
            # вычитание дельт из `w1`, `w2`, `b1` и `b2`
            scaler = 1.0 / len(bt) 

            W1 -= dW1 * scaler * lr 
            W2 -= dW2 * scaler * lr
            b1 -= db1 * scaler * lr
            b2 -= db2 * scaler * lr

        # подсчет точности:
        #   - берем  код из прошлой лабы о подсчете точности (используя `xt` и `yt`) с матрицами параметров `w1` и `w2` и байасами `b1` и `b2`
        
        print("epoch: ", epoch)

        h = sigmoid(Xt.dot(W1), False)
        p = sigmoid(h.dot(W2), False)

        print((np.sum(yt == np.argmax(p, axis=1)) / len(yt))) 


train(epochs, lr, batch, X, y, Xt, yt)
