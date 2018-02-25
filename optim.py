import numpy as np


class Optimizer:
    def step(self, *args, **kwargs):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=.9, beta2=.99, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.dx = {k: np.zeros_like(v) for k, v in params.items()}
        self.dx2 = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        for param_name, grad_value in grads.items():
            param_value = self.params[param_name]
            m = self.beta1 * self.dx[param_name] + (1 - self.beta1) * grad_value
            v = self.beta2 * self.dx2[param_name] + (1 - self.beta2) * (grad_value * grad_value)
            self.dx[param_name] = m
            self.dx2[param_name] = v
            self.params[param_name] = param_value - self.lr * m / np.sqrt(v + self.eps)
        return self.params
