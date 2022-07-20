def ad_op(func):
    def autodiff_func (*args):
        value, grad_fns = func(*args)

        grad = [[variable, grad_fn(*args)] for variable, grad_fn in zip(args, grad_fns)]
        return Variable(value, grad=grad)

    return autodiff_func
