import numpy as np

import theano as th
import theano.tensor.nnet.conv as nnconv
from theano.tensor.signal import downsample
from layer import Layer, activation_by_name

from weights import init_wb, borrow

float_x = th.config.floatX
# ############################## ConvPool Layer ################################


class ConvPoolLayer(Layer):
    def __init__(self, inpt, wts, rand_gen,
                 batch_sz, num_prev_maps, in_sz,
                 num_maps, filter_sz, stride, pool_sz,
                 actvn='tanh'):
        assert (wts is not None or rand_gen is not None)
        image_shape = (batch_sz, num_prev_maps, in_sz, in_sz)
        filter_shape = (num_maps, num_prev_maps, filter_sz, filter_sz)

        # Assign Weights
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_sz)
        self.W, self.b = init_wb(wts, rand_gen,
                                  filter_shape, (filter_shape[0], ),
                                  fan_in, fan_out, actvn, 'Conv')

        # Add ConvPool Operation to the graph
        conv_out = nnconv.conv2d(inpt, self.W, image_shape, filter_shape,
                                 subsample=(stride, stride))
        conv_pool = downsample.max_pool_2d(conv_out, (pool_sz, pool_sz),
                                           ignore_border=True)
        self.output = activation_by_name(actvn)(
            conv_pool + self.b.dimshuffle('x', 0, 'x', 'x'))

        # Calculate output shape
        self.out_sz = (in_sz - filter_sz + 1) / (stride * pool_sz)

        # Store Parameters
        self.params = [self.W, self.b]
        self.inpt = inpt
        self.num_maps = num_maps
        self.n_out = num_maps * self.out_sz ** 2
        self.args = (batch_sz, num_prev_maps, in_sz,
                     num_maps, filter_sz, stride, pool_sz, actvn)
        self.representation = ('ConvPool Maps:{:2d} Filter:{} Stride:{} Pool:{}'
                               ' Output:{:2d} Act:{}'.format(
            num_maps, filter_sz, stride, pool_sz, self.out_sz, actvn))

    def TestVersion(self, inpt):
        return ConvPoolLayer(inpt, (self.W, self.b), None, *self.args)