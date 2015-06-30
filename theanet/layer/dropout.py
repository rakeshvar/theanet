import theano as th
import theano.tensor as tt
from .layer import Layer

float_x = th.config.floatX
############################### Hidden Layer  ##################################


def drop_output(output, pdrop, rand_gen=None):
    srs = tt.shared_randomstreams.RandomStreams(rand_gen.randint(1e6)
                                                if rand_gen else None)
    mask = srs.binomial(n=1, p=1 - pdrop, size=output.shape)
    return output * tt.cast(mask, float_x)

class DropOutLayer(Layer):
    def __init__(self, inpt, rand_gen=None, n_in=None, pdrop=0):
        if pdrop:
            self.output = drop_output(inpt, pdrop, rand_gen)
        else:
            self.output = inpt
        self.inpt = inpt
        self.params = []
        self.n_in, self.n_out = n_in, n_in
        self.pdrop = pdrop

        self.representation = "Drop:{:.0%} Out:{:3d}".format(pdrop, n_in)

    def TestVersion(self, inpt):
        test_version = DropOutLayer(inpt, n_in=self.n_in, pdrop=0)
        test_version.output *= 1 - self.pdrop
        return test_version