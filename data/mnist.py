import os
import gzip
import pickle
import numpy as np


def _print_info(name, data_set):
    x, y = data_set
    print("""{}
  X::
    shape:{}
    min:{} mean:{:5.2f} max:{:5.2f}
  Y::
    shape:{}
    min:{} mean:{:5.2f} max:{}
    """.format(name,
               x.shape, x.min(), x.mean(), x.max(),
               y.shape, y.min(), y.mean(), y.max(),))


def _load_mnist():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(data_dir, "mnist.pkl.gz")

    print("Looking for data file: ", data_file)

    if not os.path.isfile(data_file):
        import urllib.request as url
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from: ', origin)
        url.urlretrieve(origin, data_file)

    print('Loading MNIST data')
    f = gzip.open(data_file, 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()
    f.close()


    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    testing_x, testing_y = test_set

    training_x = np.vstack((train_x, valid_x))
    training_y = np.concatenate((train_y, valid_y))

    training_x = training_x.reshape((training_x.shape[0], 1, 28, 28))
    testing_x = testing_x.reshape((testing_x.shape[0], 1, 28, 28))

    return training_x, training_y, testing_x, testing_y


training_x, training_y, testing_x, testing_y = _load_mnist()


if __name__ == '__main__':
    _print_info("Training Data Set:", (training_x, training_y))
    _print_info("Test Data Set:", (testing_x, testing_y))
