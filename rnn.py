import numpy as np

from layers import Layer, Affine, Tanh, CrossEntropy


class RNNCell(Layer):
    def forward(self, x, hidden_state_prev, params):
        assert len(x.shape) == 2
        affine_hidden, affine_input, affine_output, tanh = Affine(), Affine(), Affine(), Tanh()
        hidden_state_raw = affine_hidden(hidden_state_prev, params['h2h'], params['h2h_b'])
        hidden_state_raw += affine_input(x, params['i2h'], params['i2h_b'])
        hidden_state = tanh(hidden_state_raw)
        logits = affine_output(hidden_state, params['h2o'], params['h2o_b'])
        self.cache = (affine_hidden, affine_input, affine_output, tanh, params)
        return hidden_state, logits

    def backward(self, dnext_hidden_state, dlogits):
        affine_hidden, affine_input, affine_output, tanh, params = self.cache
        dparams = {}
        dhidden_state, dparams['h2o'], dparams['h2o_b'] = affine_output.backward(dlogits)
        dhidden_state = dhidden_state + dnext_hidden_state
        dhidden_state_raw = tanh.backward(dhidden_state)
        dhidden_state_prev, dparams['h2h'], dparams['h2h_b'] = affine_hidden.backward(dhidden_state_raw)
        dx, dparams['i2h'], dparams['i2h_b'] = affine_input.backward(dhidden_state_raw)
        return dx, dhidden_state_prev, dparams


class RNN(Layer):
    def forward(self, hidden_state, x, params):
        num_inputs = len(x)
        logits = []
        self.cache = []
        for i in range(num_inputs):
            rnn_cell = RNNCell()
            hidden_state, _logits = rnn_cell(np.expand_dims(x[i], 0), hidden_state, params)
            logits.append(_logits)
            self.cache.append(rnn_cell)
        self.cache = (self.cache, params)
        return hidden_state, logits

    def backward(self, dlogits):
        rnn_cells, params = self.cache
        dparams = {k: np.zeros_like(v) for k, v in params.items()}
        dnext_hidden_state = 0
        while len(rnn_cells) > 0:
            rnn_cell = rnn_cells.pop()
            _, dnext_hidden_state, _dparams = rnn_cell.backward(dnext_hidden_state, dlogits.pop())
            for param_name, grad_value in _dparams.items():
                dparams[param_name] += grad_value
        return dparams


def rnn_training_step(rnn, hidden_state, x, y, params):
    hidden_state, logits = rnn(hidden_state, x, params)
    dlogits = []
    loss = 0
    for i, l in enumerate(logits):
        criterion = CrossEntropy()
        loss += criterion(l, y[i])
        dlogits.append(criterion.backward())
    loss /= len(x)
    dparams = rnn.backward(dlogits)
    return loss, hidden_state, dparams


def sample(rnn, hidden_state, input, params, n=100):
    one_hot = []
    while n > 0:
        if len(input.shape) == 1:
            input = np.expand_dims(input, 0)
        hidden_state, logits = rnn(hidden_state, input, params)
        logits = logits[0].squeeze()
        probs = _logits_to_probs(logits)
        idx = np.random.choice(len(logits), p=probs)
        one_hot_char = np.zeros_like(logits)
        one_hot_char[idx] = 1
        one_hot.append(one_hot_char)
        input = one_hot_char
        n -= 1
    return np.asarray(one_hot)


def _logits_to_probs(logits):
    logits = logits.copy()
    logits -= np.max(logits)
    unnormalized_probs = np.exp(logits)
    return unnormalized_probs / np.sum(unnormalized_probs)


def init_params(vocab_size, hidden_size, std=0.01):
    init_weights = lambda size: np.random.randn(*size) * std
    init_bias = lambda num_outputs: np.zeros((num_outputs,))
    params = {
        'i2h': init_weights((vocab_size, hidden_size)),
        'h2h': init_weights((hidden_size, hidden_size)),
        'h2o': init_weights((hidden_size, vocab_size)),
        'i2h_b': init_bias(hidden_size),
        'h2h_b': init_bias(hidden_size),
        'h2o_b': init_bias(vocab_size)
    }
    return params
