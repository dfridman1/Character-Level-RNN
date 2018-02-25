import argparse
import numpy as np

from data_loader import DataLoader
from rnn import RNN, rnn_training_step, init_params, sample
from optim import Adam


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/input.txt')
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--seq_length', type=int, default=25)
    parser.add_argument('--sample_every', type=int, default=10000)
    parser.add_argument('--sample_size', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--print_every', type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input, 'r') as fp:
        data_loader = DataLoader(fp.read(), batch_size=args.seq_length)

    rnn = RNN()
    params = init_params(data_loader.vocab_size, hidden_size=args.hidden_size)
    optimizer = Adam(params, lr=args.lr)
    it = 0
    for epoch in range(args.num_epochs):
        hidden_state = np.zeros((1, args.hidden_size))
        for x, y in data_loader:
            if it % args.sample_every == 0:
                one_hot = sample(rnn, hidden_state, x[0], params, args.sample_size)
                generated_text = data_loader.decode(one_hot)
                print(generated_text)
            loss, hidden_state, dparams = rnn_training_step(rnn, hidden_state, x, y, params)
            if it % args.print_every == 0:
                print('iteration: {}, loss: {}'.format(it, loss))
            optimizer.step(dparams)
            it += 1


if __name__ == '__main__':
    main()
