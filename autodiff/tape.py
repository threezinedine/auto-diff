from .variable import Variable
import numpy as np


class Tape:
    def __init__(self):
        self._result_dict = {} 

    def gradient(self, source:Variable):
        def gradient_iterator(variable, value):
            if not isinstance(value, np.ndarray):
                value = np.array([[value]])

            for child, grad_val in variable.grad:
                try:
                    x, y = child.value.shape
                    if not isinstance(grad_val, np.ndarray):
                        grad_val = np.array([[grad_val]])
                    try:
#                        print(grad_val, value)
                        local_value = np.dot(grad_val.reshape(x, -1), value.reshape(-1, y))
                    except:
                        try:
                            local_value = np.dot(value.reshape(x, -1), grad_val.reshape(-1, y))
                        except:
                            local_value = value * grad_val
                    #print(local_value)
                    if child in self._result_dict.keys():
                        self._result_dict[child] += local_value
                    else:
                        self._result_dict[child] = local_value

                    gradient_iterator(child, local_value)
                except:
                    print("NOne", child.value, value, grad_val)
                    self._result_dict[child] = None

        gradient_iterator(source, 1)

    def __getitem__(self, variable:Variable):
        if variable in self._result_dict.keys():
            return self._result_dict[variable]
        else:
            return 0

    def clear(self):
        self._result_dict = {}

    def __enter__(self):
        self.clear()
        return self

    def __exit__(self, *args):
        self.clear()
