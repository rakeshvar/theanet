import theano as th
import theano.tensor as tt
from .weights import borrow

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


activation_list = [
    tt.nnet.sigmoid,
    tt.nnet.softplus,
    tt.nnet.softmax,
    Activation(lambda x: x, 'linear'),
    Activation(lambda x: 1.7*tt.tanh(2 * x / 3), 'scaled_tanh'),
    Activation(lambda x: tt.maximum(0, x), 'relu'),
    Activation(lambda x: tt.tanh(x), 'tanh'),
] + [
    Activation(lambda x, i=i: tt.maximum(0, x) + tt.minimum(0, x) * i/100,
               'relu{:02d}'.format(i))
    for i in range(100)
]

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
        raise NotImplementedError("Unknown Activation Specified: " + name)


###############################################################################

class Layer(object):
    """
    Base class for Layer
    """

    def __str__(self):
        return self.representation

    def get_wts(self):
        return [borrow(p) for p in self.params]

    def get_updates(self, cost, rate):
        if not hasattr(self, "reg"):
            return []

        if not self.reg['rate']:
            return []

        updates = []
        for param in self.params:
            update = th.shared(borrow(param) * 0.,
                               broadcastable=param.broadcastable)
            updated_update = self.reg['momentum'] * update + \
                             (1. - self.reg['momentum']) * tt.grad(cost, param)
            updates.append((update, updated_update))

            updated_param = param - self.reg['rate'] * rate * update

            if borrow(param).ndim == 2 and self.reg['maxnorm']:
                col_norms = tt.sqrt(tt.sum(tt.sqr(updated_param), axis=0))
                desired_norms = tt.clip(col_norms, 0, self.reg['maxnorm'])
                scale = (1e-7 + desired_norms) / (1e-7 + col_norms)
                updates.append((param, updated_param * scale))

            else:
                updates.append((param, updated_param))

        return updates

    def get_wtcost(self):
        try:
            return self.reg['L1'] * \
                   sum(abs(t).sum() for t in self.params) + \
                   self.reg['L2'] * \
                   sum((t**2).sum() for t in self.params)

        except AttributeError:
            return 0
