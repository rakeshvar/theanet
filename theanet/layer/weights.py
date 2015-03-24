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


def init_wb(wb, rand_gen,
            size_w, size_b,
            fan_in, fan_out,
            actvn,
            name):
    """
    Initialize the weights for any layer
    :param wb:      Could be None, numpy array or another shared variable
    :param rand_gen: Needed if wts is None to generate new weights
    :param size_w:     - do -
    :param size_b:     - do -
    :param fan_in:    - do -
    :param fan_out:   - do -
    :param actvn:     - do -
    :return:
    """
    if wb is None:
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        w_values = np.asarray(
            rand_gen.uniform(low=-w_bound, high=w_bound, size=size_w),
            dtype=float_x)
        b_values = np.zeros(size_b, dtype=float_x)

        if actvn == 'sigmoid':
            w_values *= 4
        if actvn == 'softplus' or actvn.startswith('relu'):
            b_values += 1

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


def borrow(sharedvar, boro=True):
    """
    Gets the numpy ndarray underlying a sharedVariable
    """
    return sharedvar.get_value(borrow=boro)