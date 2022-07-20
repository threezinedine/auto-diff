import numpy as np
from .ops import ad_op


class Variable:
    def __init__(self, value:np.array, grad=[], name=""):
        self.value = value
        self.grad = grad
        self._name = name
        self._index = 0

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    def __repr__(self):
        return f"Value: {self.value}"

    def __add__(self, b):
        return add(self, b)

    def __sub__(self, b):
        return subtract(self, b)

    def __mul__(self, b):
        return multiply(self, b)


def ad_op(func):
    def autodiff_func (*args):
        value, grad_fns = func(*args)
        grad = [[variable, grad_fn(*args)] for variable, grad_fn in zip(args, grad_fns)]
        return Variable(value, grad=grad)

    return autodiff_func

@ad_op
def add(a:Variable, b:Variable):
    value = a.value + b.value
    grad_fns = [
        lambda x, y: 1,
        lambda x, y: 1
    ]
    return value, grad_fns

@ad_op
def subtract(a:Variable, b:Variable):
    value = a.value - b.value
    grad_fns = [
        lambda x, y: 1,
        lambda x, y: -1
    ]
    return value, grad_fns

@ad_op
def multiply(a:Variable, b:Variable):
    value = a.value * b.value
    grad_fns = [
        lambda x, y: y.value,
        lambda x, y: x.value
    ]
    return value, grad_fns

@ad_op
def negative(a:Variable):
    value = -a.value
    grad_fns = [
        lambda x: -1
    ]
    return value, grad_fns

@ad_op
def mat_mul(a:Variable, b:Variable):
    value = np.dot(a.value, b.value)
    grad_fns = [
        lambda x, y: y.value,
        lambda x, y: x.value
    ]
    return value, grad_fns

@ad_op
def average(a:Variable):
    value = np.average(a.value)
    sample = a.value.shape[0]
    result = np.ones_like(a.value) / sample
    grad_fns = [
        lambda x: result
    ]
    return value, grad_fns

@ad_op
def square(a:Variable):
    value = np.square(a.value)
    grad_fns = [
        lambda x: 2 * a.value
    ]
    return value, grad_fns

@ad_op
def transpose(a:Variable):
    value = np.transpose(a.value)
    grad_fns = [
        lambda x: 1
    ]
    return value, grad_fns

def sigmoid_fn(value):
    return 1/(1 +np.exp(value))

@ad_op
def sigmoid(a:Variable):
    value = sigmoid_fn(a.value)
    grad_fns = [
        lambda x: sigmoid_fn(x.value) * (1 - sigmoid_fn(x.value)) 
    ]
    return value, grad_fns


@ad_op
def relu(a:Variable):
    res = np.where(a.value >= 0, a.value, 0)
    def grad_fun(a):
        return np.where(a.value > 0, 1, 0)
    return res, [grad_fun]
