import numpy as np

import theano as th
import theano.tensor as tt
import theano.tensor.signal.conv as sigconv
from .layer import Layer

float_x = th.config.floatX
# ############################## Input Layer  ##################################


class InputLayer(Layer):
    def __init__(self, inpt, img_sz, num_maps=1, rand_gen=None):
        self.params = []
        self.inpt = inpt
        self.out_sz = img_sz
        self.num_maps = num_maps
        self.n_out = self.num_maps * self.out_sz ** 2
        self.representation = \
            'Input Maps:{} Sizes Input:{:2d} Output:{:2d}'.format(num_maps,
                                                                  img_sz,
                                                                  img_sz)
        self.output = inpt

    def TestVersion(self, inpt):
        return InputLayer(inpt, self.out_sz, self.num_maps)


class ElasticLayer(Layer):
    def __init__(self, inpt, img_sz,
                 num_maps = 1,
                 translation=0,
                 zoom=1,
                 magnitude=0,
                 sigma=1,
                 pflip=0,
                 angle=0,
                 rand_gen=None,
                 invert_image=False,
                 nearest=False):
        self.inpt = inpt
        self.img_sz = img_sz
        self.translation = translation
        self.zoom = zoom
        self.magnitude = magnitude
        self.sigma = sigma
        self.invert = invert_image
        self.nearest = nearest

        self.out_sz = img_sz
        self.num_maps = num_maps
        self.n_out = self.num_maps * self.out_sz ** 2
        self.params = []
        self.representation = ('Elastic Maps:{:d} Size:{:2d} Translation:{:} '
                               'Zoom:{} Mag:{:d} Sig:{:d} Noise:{} '
                               'Angle:{} Invert:{} '
                               'Interpolation:{}'.format(
            self.num_maps, img_sz,
            translation, zoom, magnitude, sigma,
            pflip, angle, invert_image,
            'Nearest' if nearest else 'Linear'))

        if invert_image:
            inpt = 1 - inpt

        assert zoom > 0
        if not (magnitude or translation or pflip or angle) and zoom == 1:
            self.output = inpt
            self.debugout = [self.output, tt.as_tensor_variable((0, 0))]
            return

        srs = tt.shared_randomstreams.RandomStreams(rand_gen.randint(1e6)
                                                    if rand_gen else None)
        h = w = img_sz

        # Humble as-is beginning
        target = tt.as_tensor_variable(np.indices((h, w)))

        # Translate
        if translation:
            transln = translation * srs.uniform((2, 1, 1), -1)
            target += transln

        # Apply elastic transform
        if magnitude:
            # Build a gaussian filter
            var = sigma ** 2
            filt = np.array([[np.exp(-.5 * (i * i + j * j) / var)
                             for i in range(-sigma, sigma + 1)]
                             for j in range(-sigma, sigma + 1)], dtype=float_x)
            filt /= 2 * np.pi * var

            # Elastic
            elast = magnitude * srs.normal((2, h, w))
            elast = sigconv.conv2d(elast, filt, (2, h, w), filt.shape, 'full')
            elast = elast[:, sigma:h + sigma, sigma:w + sigma]
            target += elast

        # Center at 'about' half way
        if zoom-1 or angle:
            origin = srs.uniform((2, 1, 1), .25, .75) * \
                     np.array((h, w)).reshape((2, 1, 1))
            target -= origin

            # Zoom
            if zoom-1:
                zoomer = tt.exp(np.log(zoom) * srs.uniform((2, 1, 1), -1))
                target *= zoomer

            # Rotate
            if angle:
                theta = angle * np.pi / 180 * srs.uniform(low=-1)
                c, s = tt.cos(theta), tt.sin(theta)
                rotate = tt.stack(c, -s, s, c).reshape((2,2))
                target = tt.tensordot(rotate, target, axes=((0, 0)))

            # Uncenter
            target += origin

        # Clip the mapping to valid range and linearly interpolate
        transy = tt.clip(target[0], 0, h - 1 - .001)
        transx = tt.clip(target[1], 0, w - 1 - .001)

        if nearest:
            vert = tt.iround(transy)
            horz = tt.iround(transx)
            output = inpt[:, :, vert, horz]
        else:
            topp = tt.cast(transy, 'int32')
            left = tt.cast(transx, 'int32')
            fraction_y = tt.cast(transy - topp, float_x)
            fraction_x = tt.cast(transx - left, float_x)

            output = inpt[:, :, topp, left] * (1 - fraction_y) * (1 - fraction_x) + \
                     inpt[:, :, topp, left + 1] * (1 - fraction_y) * fraction_x + \
                     inpt[:, :, topp + 1, left] * fraction_y * (1 - fraction_x) + \
                     inpt[:, :, topp + 1, left + 1] * fraction_y * fraction_x

        # Now add some noise
        if pflip:
            mask = srs.binomial(n=1, p=pflip, size=inpt.shape, dtype=float_x)
            output = (1 - output) * mask + output * (1 - mask)

        self.output = output
        self.debugout = [self.output,
                         target - np.indices((h, w)),]

        if translation:
            self.debugout.append(transln)
        if zoom-1 or angle:
            self.debugout.append(origin)
        if angle:
            self.debugout.append(theta*180/np.pi)
        if zoom-1:
            self.debugout.append(zoomer)

    def TestVersion(self, te_inpt):
        return ElasticLayer(te_inpt, self.img_sz,
                            translation=0, zoom=1,
                            magnitude=0, sigma=1,
                            pflip=0, angle=0,
                            invert_image=self.invert,
                            nearest=self.nearest)