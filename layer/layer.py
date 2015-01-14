import theano as th
import theano.tensor as tt
from weights import borrow

float_x = th.config.floatX
###############################################################################
#   A bunch of Activations
###############################################################################


class Activation:
    """
    Defines a bunch of activations as callable classes.
    Useful for printing and specifying activations as strings.
    """
    def __init__(self, fn, name):
        self.fn = fn
        self.name = name

    def __call__(self, *args):
        return self.fn(*args)

    def __str__(self):
        return self.name


scaled_tanh = Activation(lambda x: 1.7 * tt.tanh(2 * x / 3), 'scaled_tanh')
tanh = Activation(lambda x: tt.tanh(x), 'tanh')
relu = Activation(lambda x: tt.maximum(0, x), 'relu')
linear = Activation(lambda x: x, 'linear')

activation_list = (scaled_tanh, relu, linear,
                   tt.nnet.sigmoid, tanh,
                   tt.nnet.softplus, tt.nnet.softmax)


def activation_by_name(name):
    """
    Get an activation function or callabe-class from its name
    :param name: string
    :return: Callable Activation
    """
    for act in activation_list:
        if name == str(act):
            return act
    else:
        raise NotImplementedError


###############################################################################
#   A bunch of Layers
###############################################################################

class Layer(object):
    """
    Base class for Layer
    """

    def __str__(self):
        return self.representation

    def get_wts(self):
        return map(borrow, self.params)