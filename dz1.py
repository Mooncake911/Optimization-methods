import pandas as pd
import random
import math
import time

e = 10**-6
input_data = pd.read_csv("tables/input.csv", sep=',')
input_x = input_data['X'][0:]
input_y = input_data['Y'][0:]

# Choose random parameters
a0 = random.randint(-10, 10)
b0 = random.randint(-10, 10)
g1 = random.random()
g2 = math.sqrt(1 - g1 ** 2)
print("Using parameters: a0=%d, b0=%d , g1=%f , g2=%f" % (a0, b0, g1, g2))


def functions(a, b):
    sigma1 = 0
    sigma2 = 0
    for i in range(len(input_x)):
        sigma1 += pow(a*input_x[i] + b - input_y[i], 2)     # First function F1
        sigma2 += abs(a*input_x[i] + b - input_y[i])        # Second function F2
    return sigma1, sigma2


def dichotomy(p1, p2):
    table = []
    # Searching minimum in two functions
    for i in range(2):  # F(i)
        a = p1
        b = p2
        t = e/2
        while abs(b - a) > e:
            c = (a+b)/2
            F_tc = functions(a0 + (c-t)*g1, b0 + (c-t)*g2)[i]
            F_ct = functions(a0 + (c+t)*g1, b0 + (c+t)*g2)[i]
            table.append([a, b, c, (b-a)/2, F_tc, F_ct])
            if F_tc < F_ct:
                b = c - t
            else:
                a = c + t
        table.append([None, None, None, None, None, None])
    return table


def golden_ratio(p1, p2):
    table = []
    t = 0.618033    # Golden cross
    # Searching minimum in two functions
    for i in range(2):  # F(i)
        a = p1
        b = p2
        c = a + (1 - t) * (b - a)
        d = a + t * (b - a)
        F_c = functions(a0 + c * g1, b0 + c * g2)[i]
        F_d = functions(a0 + d * g1, b0 + d * g2)[i]
        while abs(b - a) > e:
            table.append([a, b, c, d, (b-a)/2, F_c, F_d])
            if F_c > F_d:
                a = c
                c = d
                F_c = F_d
                d = a + t * (b - a)
                F_d = functions(a0 + d * g1, b0 + d * g2)[i]
            else:
                b = d
                d = c
                F_d = F_c
                c = a + (1 - t) * (b - a)
                F_c = functions(a0 + c * g1, b0 + c * g2)[i]
        table.append([None, None, None, None, None, None, None])
    return table


def main():
    start = time.time()
    table1 = dichotomy(-100, 100)
    print("dichotomy time: ", time.time() - start)
    start = time.time()
    table2 = golden_ratio(-100, 100)
    print("golden_ratio time: ", time.time() - start)

    # Dichotomy
    data = pd.DataFrame(table1)
    data.columns = ['a', 'b', 'c', '(b-a)/2', 'F(c-e)', 'F(c+e)']
    data.to_csv("tables/dichotomy.csv")

    # Golden_ratio
    data = pd.DataFrame(table2)
    data.columns = ['a', 'b', 'c', 'd', '(b-a)/2', 'F(c)', 'F(d)']
    data.to_csv("tables/golden_ratio.csv")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
