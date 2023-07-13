import pandas as pd
import time
from scipy.optimize import minimize_scalar
import numpy as np

e = 10 ** -6
input_data = pd.read_csv("tables/input.csv", sep=',')
input_x = input_data['X'][0:]
input_y = input_data['Y'][0:]


def function(a, b, id_fun):
    sigma = 0
    # First function F1
    if id_fun == 0:
        for i in range(len(input_x)):
            sigma += pow(a * input_x[i] + b - input_y[i], 2)
    # Second function F2
    elif id_fun == 1:
        for i in range(len(input_x)):
            sigma += abs(a * input_x[i] + b - input_y[i])
    return sigma


def grad_function(a, b, id_fun):
    sigma_a = 0
    sigma_b = 0
    # First function F1
    if id_fun == 0:
        for i in range(len(input_x)):
            sigma_a += 2 * input_x[i] * (a * input_x[i] + b - input_y[i])
            sigma_b += 2 * a * input_x[i] + 2 * b - 2 * input_y[i]
    return np.array([sigma_a, sigma_b])


def coord_descent(start_coords):
    """ Метод по координатного спуска """
    data = []
    for id_fun in range(2):
        coords = np.array(start_coords)
        new = function(*coords, id_fun)
        while True:
            old = new
            old_coords = coords.copy()

            # Минимизация по x1
            f = lambda x1: function(x1, coords[1], id_fun)
            res = minimize_scalar(f)  # одномерная оптимизация по переменной (метод Брента)
            coords[0] = res.x
            # Минимизация по x2
            f = lambda x2: function(coords[0], x2, id_fun)
            res = minimize_scalar(f)  # одномерная оптимизация по переменной (метод Брента)
            coords[1] = res.x
            # Запоминаем новое значение функции
            new = function(*coords, id_fun)

            data.append([coords, np.linalg.norm(np.abs(coords - old_coords)), new, abs(new - old)])
            if abs(new - old) < e:
                break

        data.append([None, None, None, None])
    return pd.DataFrame(data)


def gradient_with_step(start_coords, alpha=0.3, beta=0.5):
    """ Градиентный метод с дроблением шага """
    data = []
    for id_fun in range(1):
        coords = np.array(start_coords)
        grad = e + 1
        while np.linalg.norm(grad) > e:
            f = function(*coords, id_fun)
            grad = grad_function(*coords, id_fun)
            data.append([coords, f, np.linalg.norm(grad)])

            step = alpha
            while function(*(coords - step * grad), id_fun) >= f - beta * step * (np.linalg.norm(grad) ** 2):
                step *= beta
            coords = coords - step * grad
        data.append([None, None, None])
    return pd.DataFrame(data)


def steepest_descent(start_coords):
    """ Метод наискорейшего градиентного спуска """
    data = []
    for id_fun in range(1):
        coords = np.array(start_coords)
        grad = e + 1
        while np.linalg.norm(grad) > e:
            grad = grad_function(*coords, id_fun)
            data.append([coords, function(*coords, id_fun), np.linalg.norm(grad)])

            alpha = minimize_scalar(lambda a: function(*(coords - a * grad), id_fun)).x  # (метод Брента)
            coords = coords - alpha * grad

        data.append([None, None, None])
    return pd.DataFrame(data)


def gradient_with_fixed_step(start_coords):
    """ Градиентный метод с фиксированным шагом """
    data = []
    for id_fun in range(1):
        L = 17.6
        coords = np.array(start_coords)
        grad = e + 1
        while np.linalg.norm(grad) > e:
            grad = grad_function(*coords, id_fun)
            data.append([coords, function(*coords, id_fun), np.linalg.norm(grad)])
            coords = coords - (1 / L) * grad

        data.append([None, None, None])
    return pd.DataFrame(data)


def main():
    start = time.time()
    table1 = coord_descent(start_coords=[10000, 10000])
    table1.columns = ['Xn', '||Xn - Xn-1||', 'F(Xn)', '|F(Xn) - F(Xn-1)|']
    table1.to_csv("tables/coord_descent.csv")
    print("coord_descent: ", time.time() - start)

    start = time.time()
    table2 = gradient_with_step(start_coords=[10000, 10000])
    table2.columns = ['Xn', 'F(Xn)', '||F`(Xn)||']
    table2.to_csv("tables/gradient_with_step.csv")
    print("gradient_with_step: ", time.time() - start)

    start = time.time()
    table3 = steepest_descent(start_coords=[10000, 10000])
    table3.columns = ['Xn', 'F(Xn)', '||F`(Xn)||']
    table3.to_csv("tables/steepest_descent.csv")
    print("steepest_descent: ", time.time() - start)

    start = time.time()
    table4 = gradient_with_fixed_step(start_coords=[10000, 10000])
    table4.columns = ['Xn', 'F(Xn)', '||F`(Xn)||']
    table4.to_csv("tables/gradient_with_fixed_step.csv")
    print("gradient_with_fixed_step: ", time.time() - start)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
