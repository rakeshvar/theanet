import numpy as np
import theano

# ##############################################################################
#   Helper Functions
###############################################################################
float_x = theano.config.floatX


def is_shared_var(x):
    """
    Checks if x is SharedVar.
    Could be a CUDA SharedVar or just a normal Theano SharedVar
    """
    return 'SharedVariable' in str(type(x))


def borrow(sharedvar, boro=True):
    """
    Gets the numpy ndarray underlying a sharedVariable
    """
    return sharedvar.get_value(borrow=boro)


def init_wb(wb, rand_gen,
            size_w, size_b,
            fan_in, fan_out,
            actvn,
            name):
    """
    Initialize the weights. If wb is given, the weights are initialized
    as a copy of the same. If wb is None, they are randomly initialized based
    on the rest of the arguments.

    :param wb: w and b (to be copied or None)
    :type wb: None or ndarray or SharedVariable

    The following a are needed only when wb is None.

    :param RandomStream rand_gen: A random stream.
    :param tuple size_w: Size of w
    :param size_b: Size of b
    :type size_b: tuple or int
    :param int fan_in: Number of units coming in.
    :param int fan_out: Number of units going out.
    :param str actvn: The activation that will be applied. See `Activation`s
    :return: The initialized weights
    :rtype: SharedVariable
    """
    if wb is None:
        if len(size_w) == 4:
            w_values = 2. * rand_gen.randint(2, size=size_w) - 1
            #w_values = rand_gen.uniform(low=-1, high=1, size=size_w)
            w_values /= np.sqrt(fan_in)
        else:
            w_values = rand_gen.uniform(low=-1, high=1, size=size_w)
            w_values *= np.sqrt(6 / (fan_in + fan_out))

        w_values = np.asarray(w_values, dtype=float_x)
        b_values = np.zeros(size_b, dtype=float_x)

        if actvn == 'sigmoid':
            w_values *= 4
        if actvn in ('softplus', 'relu') or actvn.startswith('relu0'):
            b_values += .5

    elif type(wb[0]) is np.ndarray:
        w_values, b_values = wb

    else:
        assert is_shared_var(wb[0])

    if wb and is_shared_var(wb[0]):
        # Usually true when getting TestVersion
        w, b = wb

    else:
        w = theano.shared(w_values, name=name+'W', borrow=True)
        b = theano.shared(b_values, name=name+'b', borrow=True)

    return w, b
