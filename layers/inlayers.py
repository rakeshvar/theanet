import numpy as np

import theano as th
import theano.tensor as tt
import theano.tensor.signal.conv as sigconv
from layers.layer import Layer

float_x = th.config.floatX
# ############################## Input Layer  ##################################


class InputLayer(Layer):
    def __init__(self, inpt, img_sz, ):
        self.params = []
        self.inpt = inpt
        self.out_sz = img_sz
        self.num_maps = 1
        self.n_out = self.num_maps * self.out_sz ** 2
        self.representation = \
            'Input Maps:1 Sizes Input:{:2d} Output:{:2d}'.format(img_sz, img_sz)

    def TestVersion(self, inpt):
        return InputLayer(inpt, self.out_sz)


class ElasticLayer(Layer):
    def __init__(self, inpt, img_sz, translation, zoom, magnitude, sigma,
                 pflip, rand_gen=None):
        self.inpt = inpt
        self.img_sz = img_sz
        self.translation = translation
        self.zoom = zoom
        self.magnitude = magnitude
        self.sigma = sigma

        self.out_sz = img_sz
        self.num_maps = 1
        self.n_out = self.num_maps * self.out_sz ** 2
        self.params = []
        self.representation = ('Elastic Maps:{:d} Size:{:2d} Translation:{:} '
                               'Zoom:{} Mag:{:2d} Sig:{:2d}'.format(
            self.num_maps, img_sz, translation, zoom, magnitude, sigma))

        assert zoom > 0
        if magnitude + translation + pflip == 0 and zoom == 1:
            self.output = inpt
            return

        srs = tt.shared_randomstreams.RandomStreams(rand_gen.randint(1e6)
                                                    if rand_gen else None)
        h = w = img_sz

        # Build a gaussian filter
        var = sigma ** 2
        filt = np.array([[np.exp(-.5 * (i * i + j * j) / var)
                          for i in range(-sigma, sigma + 1)]
                         for j in range(-sigma, sigma + 1)], dtype=float_x)
        filt /= 2 * np.pi * var

        # Build a transition grid as a translation + elastic distortion
        trans = magnitude * srs.normal((2, h, w))
        trans += translation * srs.uniform((2, 1, 1), -1)
        trans = sigconv.conv2d(trans, filt, (2, h, w), filt.shape, 'full')
        trans = trans[:, sigma:h + sigma, sigma:w + sigma]
        trans += np.indices((h, w))

        # Now zoom it
        halves = np.array((h // 2, w // 2)).reshape((2, 1, 1))
        zoomer = tt.exp(np.log(zoom) * srs.uniform((2, 1, 1), -1))
        trans -= halves
        trans *= zoomer
        trans += halves

        # Clip the mapping to valid range and linearly interpolate
        transy = tt.clip(trans[0], 0, h - 1 - .001)
        transx = tt.clip(trans[1], 0, w - 1 - .001)
        topp = tt.cast(transy, 'int32')
        left = tt.cast(transx, 'int32')
        fraction_y = tt.cast(transy - topp, float_x)
        fraction_x = tt.cast(transx - left, float_x)
        self.trans = trans

        output = inpt[:, topp, left] * (1 - fraction_y) * (1 - fraction_x) + \
                 inpt[:, topp, left + 1] * (1 - fraction_y) * fraction_x + \
                 inpt[:, topp + 1, left] * fraction_y * (1 - fraction_x) + \
                 inpt[:, topp + 1, left + 1] * fraction_y * fraction_x

        # Now add some noise
        mask = srs.binomial(n=1, p=pflip, size=inpt.shape, dtype=float_x)
        self.output = (1 - output) * mask + output * (1 - mask)

    def TestVersion(self, te_inpt):
        return ElasticLayer(te_inpt, self.img_sz, 0, 1, 0, 0, 0)