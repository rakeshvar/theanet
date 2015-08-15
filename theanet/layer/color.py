import numpy as np
import theano as th
import theano.tensor as tt
from .layer import Layer

float_x = th.config.floatX


class ColorLayer(Layer):
    def __init__(self, inpt, img_sz,
                 num_maps=3,
                 rand_gen=None,
                 balance=1,
                 gamma=1,
                 maxval=1):
        self.params = []
        self.inpt = inpt
        self.out_sz = img_sz
        self.num_maps = num_maps
        self.n_out = self.num_maps * self.out_sz ** 2
        self.representation = 'Color Maps:{} Size:{:2d} Balance:{:.2f} ' \
                              'Gamma:{:.2f} Maxval:{}'.format(
            num_maps, img_sz, balance, gamma, maxval)

        if gamma == 1 and balance == 1:
            self.output = inpt
            return

        assert gamma > 0 and balance > 0
        srs = tt.shared_randomstreams.RandomStreams(rand_gen.randint(1e6)
                                                    if rand_gen else None)
        def pos_rand(a):
            return tt.exp(np.log(a) * srs.uniform((inpt.shape[0], num_maps), -1)
            ).dimshuffle(0, 1, 'x', 'x').astype(float_x)


        out = inpt / maxval
        out *= pos_rand(balance)
        out = tt.clip(out, 0,  1)
        out **= pos_rand(gamma)
        out = 1 - (1-out) ** pos_rand(gamma)

        self.output = out * maxval

    def TestVersion(self, inpt):
        return ColorLayer(inpt,
                          self.out_sz,
                          num_maps=self.num_maps,
                          rand_gen=None,
                          balance=1,
                          gamma=1,
                          maxval=1)
