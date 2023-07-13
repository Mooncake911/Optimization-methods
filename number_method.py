import pandas as pd

input_data = pd.read_csv("tables/gg.csv", sep=',')
input_x = input_data['x'][0:]
input_y = input_data['y'][0:]
h = 0.05

input = list(input_y)
k = 0
while len(input) > 1:
    list = []
    for i in range(len(input) - 1):
        list.append(float(input[i + 1] - input[i]))
    input = list.copy()
    while len(list) < len(input_y):
        list.append(None)
    k += 1
    name = 'delta{0}(y)'.format(k)
    input_data[name] = list

print(input_data)
