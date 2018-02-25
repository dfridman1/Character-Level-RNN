import numpy as np


class Layer:
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Affine(Layer):
    def forward(self, x, w, b):
        self.cache = (x, w, b)
        return x.dot(w) + b

    def backward(self, dout):
        x, w, b = self.cache
        db = dout.sum(axis=0)
        dw = x.T.dot(dout)
        dx = dout.dot(w.T)
        return dx, dw, db


class Tanh(Layer):
    def forward(self, x):
        out = np.tanh(x)
        self.cache = out
        return out

    def backward(self, dout):
        out = self.cache
        return (1 - out * out) * dout


class CrossEntropy(Layer):
    def forward(self, logits, target):
        if len(logits.shape) == 1:
            logits = np.expand_dims(logits, 0)
        if len(target.shape) == 1:
            target = np.expand_dims(target, 0)
        target = np.argmax(target, axis=1)
        logits = logits.copy()
        logits -= np.max(logits, axis=1)
        unnormalized_probs = np.exp(logits)
        probs = unnormalized_probs / np.sum(unnormalized_probs, axis=1, keepdims=True)
        correct_class_probs = probs[np.arange(len(logits)), target]
        self.cache = (probs, target)
        return np.mean(-np.log(correct_class_probs))

    def backward(self):
        probs, target = self.cache
        dlogits = probs.copy()
        dlogits[np.arange(len(dlogits)), target] -= 1
        return dlogits / dlogits.shape[0]
