import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import math

# задаваемая точность
e = 10 ** (-6)


# -------------------------------------------------------------------------------------------------------------------- #


def function_1(x):
    a, b, c, d, h = 3, 1, 5, 12, -1
    return (1 / d) * abs(a - x) * (0.5 * abs(x + b) + h) * (3 * abs(0.25 * x + c) + 1)


def Polyline_method(diapason):
    """ МЕТОД ЛОМАНЫХ """

    def Lipshitz():
        """С начала находим константу Липшица"""
        a = diapason[0]
        b = a + e
        c = -10 ** 100
        while b < diapason[1]:
            c = max(c, abs(function_1(b) - function_1(a)) / (b - a))
            a += e
            b += e
        return c

    L = 17.635415182279978
    print("Константа Липшица: {0}\n".format(L))

    list_for_data = []
    mas = []

    x0 = 1 / (2 * L) * (function_1(diapason[0]) - function_1(diapason[1]) + L * (diapason[0] + diapason[1]))
    p0 = 1 / 2 * (function_1(diapason[0]) + function_1(diapason[1]) + L * (diapason[0] - diapason[1]))
    DELTA = 1
    mas.append([p0, x0])

    while 2 * L * DELTA > e:
        p0, x0 = min(mas)
        mas.remove(min(mas))

        DELTA = 1 / (2 * L) * (function_1(x0) - p0)
        p = (1 / 2) * (function_1(x0) + p0)
        x1 = x0 - DELTA
        x2 = x0 + DELTA
        mas.append([p, x1])
        mas.append([p, x2])

        list_for_data.append([x0, p0, 2 * L * DELTA, x1, x2, p])

    # Вывод графика
    y: [float] = lambda i: function_1(i)
    x = np.linspace(diapason[0], diapason[1], 100)
    plt.plot(x, y(x))
    for x0, p0 in [sublist[:2] for sublist in list_for_data]:
        plt.plot(x0, p0, 'ro')
    plt.title("Метод ломаных")
    plt.show()

    data = pd.DataFrame(list_for_data)
    return data


# -------------------------------------------------------------------------------------------------------------------- #


def function_2(x):
    a, b, c, d = -4, -4, 3, -1
    return ((x - a) * 2 + b) * ((x + c) * 2 - d)


def derivative(x):
    a, b, c, d = -4, -4, 3, -1
    return 2 * (-2 * a + b + 2 * c - d + 4 * x)


def derivative2(x):
    return 8


def NewtonRaphson(diapason):
    """ МЕТОД: Ньютона–Рафсона """
    list_for_data = []
    x = diapason[1]
    while abs(derivative(x)) > e ** 2:
        DELTA = math.sqrt(derivative(x)**2 / derivative2(x))
        if DELTA <= (1 / 4):
            a = 1
        else:
            a = 1 / (1 + DELTA)
        x -= a * derivative(x) / derivative2(x)
        list_for_data.append([x, derivative(x), derivative2(x), a])
    data = pd.DataFrame(list_for_data)

    # Вывод графика
    y: [float] = lambda i: function_2(i)
    x = np.linspace(diapason[0], diapason[1], 100)
    plt.plot(x, y(x))
    for x, in [sublist[:1] for sublist in list_for_data]:
        plt.plot(x, y(x), 'ro')
    plt.title("Метод Ньютона–Рафсона")
    plt.show()

    return data


# -------------------------------------------------------------------------------------------------------------------- #

def main():
    df = Polyline_method([-10, 10])
    df.columns = ["x0", "p0", "2*L*DELTA", "x1", "x2", "p"]
    print(tabulate(df, headers='keys', tablefmt='psql'))
    df = NewtonRaphson([-10, 10])
    df.columns = ["x", "f'", "f''", "a"]
    print(tabulate(df, headers='keys', tablefmt='psql'))


if __name__ == '__main__':
    main()
