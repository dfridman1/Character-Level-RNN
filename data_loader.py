import numpy as np


class DataLoader(object):
    def __init__(self, data, batch_size=8):
        assert isinstance(data, str) and len(data) > 1
        assert 0 < batch_size < len(data)
        self._data = self._preprocess_data(data)
        self._batch_size = batch_size
        self.curr_pos = 0

        all_chars = sorted(set(self._data))
        self._char_to_index = {ch: idx for idx, ch in enumerate(all_chars)}
        self._index_to_char = {idx: ch for ch, idx in self._char_to_index.items()}

    @property
    def num_examples(self):
        return len(self._data) - 1

    @property
    def vocab_size(self):
        return len(self._char_to_index)

    @property
    def vocab(self):
        return set(self._char_to_index.keys())

    def next_batch(self):
        data = self._data[self.curr_pos:self.curr_pos + self._batch_size + 1]
        self.curr_pos += self._batch_size
        if self.curr_pos >= self.num_examples:
            self.curr_pos = 0
        data = self._one_hot(data)
        return data[:-1, :], data[1:, :]

    def _one_hot(self, s):
        one_hot = np.zeros((len(s), self.vocab_size))
        idx = [self._char_to_index[ch] for ch in s]
        one_hot[np.arange(len(s)), idx] = 1
        return one_hot

    def decode(self, one_hot):
        if len(one_hot.shape) == 1:
            one_hot = one_hot.reshape(1, -1)
        return ''.join(map(self.decode_char, one_hot))

    def decode_char(self, one_hot):
        idx = np.argmax(one_hot)
        return self._index_to_char[idx]

    def __iter__(self):
        while True:
            yield next(self)
            if self.curr_pos == 0:
                break

    def __next__(self):
        return self.next_batch()

    def _preprocess_data(self, data):
        return data.lower()
