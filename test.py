import autodiff as ad
from autodiff import *
import numpy as np
import matplotlib.pyplot as plt

dtype = np.float32

input_data = np.array([-1, 0, 1, 2, 3, 4, 5])
output_data = np.array([1, 0, 1, 4, 9, 16, 25]) + 2

plt.scatter(input_data, output_data, color='red')

min_data_x = np.min(input_data)
max_data_x = np.max(input_data)

min_data_y = np.min(output_data)
max_data_y = np.max(output_data)

input_data = input_data / (max_data_x - min_data_x)
output_data = output_data / (max_data_y - min_data_y)

input_arr = [Variable(np.array([[input_point, input_point ** 2]], dtype=np.float32)) for input_point in input_data]
expected_arr = [Variable(np.array([[output_point]], dtype=np.float32)) for output_point in output_data]

b = Variable(np.array([[0.5, 0.5], [0.5, .5]], dtype=dtype), name="Weight")
c = Variable(np.array([[.2, 0.]], dtype=dtype), name="Bias 1")
d = Variable(np.array([[.3], [.2]], dtype=dtype), name="Weight")
e = Variable(np.array([[0.]], dtype=dtype), name="Bias")

alpha = 0.03
losss = []

def calculate(value):
    out = sigmoid_fn(value)
    return np.dot(out, d.value) + e.value


for _ in range(800):
    for input_point, expected in zip(input_arr, expected_arr):
        with Tape() as tape:
            res1 = mat_mul(input_point, b) + c
            loss = MSE()(expected, res2)

            tape.gradient(loss)

            b.value -= alpha * tape[b]
            c.value -= alpha * tape[c]

    losss.append(loss.value)


#print(b.value)
#print(c.value)
#print(d.value)
#print(e.value)
#print(losss[-1])
#
#plt.plot(losss)
#plt.show()


plot_x = np.linspace(-3, 12, 200)

plot_x_var = [Variable(np.array([[plot_point / (max_data_x - min_data_x), 
    plot_point ** 2 / (max_data_x - min_data_x) ** 2]], dtype=np.float32)) for plot_point in plot_x]
plot_y = []

for plot_x_var_con in plot_x_var:
    res1 = mat_mul(plot_x_var_con, b) + c
    out1 = relu(res1)
    res2 = mat_mul(res1, d) + e
    plot_y.append(res2.value.item() * (max_data_y - min_data_y))

plt.plot(plot_x, np.array(plot_y))
plt.show()
