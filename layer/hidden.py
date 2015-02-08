import theano as th
import theano.tensor as tt
from .layer import Layer, activation_by_name
from .weights import init_wb, borrow

float_x = th.config.floatX
############################### Hidden Layer  ##################################


def drop_output(output, pdrop, rand_gen):
    srs = tt.shared_randomstreams.RandomStreams(rand_gen.randint(1e6)
                                                if rand_gen else None)
    mask = srs.binomial(n=1, p=1 - pdrop, size=output.shape)
    return output * tt.cast(mask, float_x)


class HiddenLayer(Layer):
    def __init__(self, inpt, wts, rand_gen=None, n_in=None, n_out=None, pdrop=0,
                 actvn='softplus'):
        assert wts is not None or rand_gen is not None

        try:
            fan_in_out = n_in + n_out
        except TypeError:
            fan_in_out = None

        self.w, self.b = init_wb(wts, rand_gen, (n_in, n_out), (n_out,),
                                  fan_in_out, fan_in_out, actvn, 'Hid')
        n_in, n_out = borrow(self.w).shape

        self.output = activation_by_name(actvn)((tt.dot(inpt, self.w) + self.b))
        if pdrop:
            self.output = drop_output(self.output, pdrop, rand_gen)

        self.inpt = inpt
        self.params = [self.w, self.b]
        self.n_in, self.n_out = n_in, n_out
        self.actvn = actvn
        self.pdrop = pdrop
        self.representation = "Hidden In:{:3d} Out:{:3d} Act:{} Drop%:{}". \
            format(n_in, n_out, actvn, pdrop)

    def TestVersion(self, inpt):
        test_version = HiddenLayer(inpt, (self.w, self.b),
                                   pdrop=0,
                                   actvn=self.actvn)
        test_version.output *= 1 - self.pdrop
        return test_version