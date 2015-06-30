import numpy as np
import math
import theano as th
import theano.tensor as tt
import theano.tensor.nnet.conv as nnconv
from theano.tensor.signal import downsample
from .layer import Layer, activation_by_name

from .weights import init_wb

float_x = th.config.floatX
# ############################## ConvPool Layer ################################


class ConvLayer(Layer):
    def __init__(self, inpt, wts, rand_gen,
                 batch_sz, num_prev_maps, in_sz,
                 num_maps, filter_sz, stride,
                 mode='valid',
                 actvn='relu50',
                 reg=()):
        """

        :param inpt:
        :param wts:
        :param rand_gen:
        :param batch_sz:
        :param num_prev_maps:
        :param in_sz:
        :param num_maps:
        :param filter_sz:
        :param stride:
        :param mode: "valid", "full", "same"
        :param actvn:
        :param reg:
        :raise NotImplementedError:

        TODO: Add support for rectangular input
        """
        assert (wts is not None or rand_gen is not None)
        assert mode in ("valid", "full", "same")

        image_shape = (batch_sz, num_prev_maps, in_sz, in_sz)
        filter_shape = (num_maps, num_prev_maps, filter_sz, filter_sz)

        # Assign Weights
        fan_in = num_prev_maps * filter_sz * filter_sz
        fan_out = num_maps * filter_sz * filter_sz
        self.W, self.b = init_wb(wts, rand_gen,
                                  filter_shape, (filter_shape[0], ),
                                  fan_in, fan_out, actvn, 'Conv')

        # Add Conv2D Operation to the graph
        border_mode = "valid" if mode == "valid" else "full"
        conv_out = nnconv.conv2d(inpt, self.W, image_shape, filter_shape,
                                 subsample=(stride, stride),
                                 border_mode=border_mode)
        if mode == 'same':
            assert stride == 1, "For Same mode stride should be 1"
            shift = (filter_sz - 1) // 2
            conv_out = conv_out[:, :, shift:in_sz + shift, shift:in_sz + shift]
            self.out_sz = in_sz

        elif mode == "full":
            self.out_sz = in_sz + filter_sz + 1

        elif mode == "valid":
            self.out_sz = in_sz - filter_sz + 1

        # TODO: Remove stride support OR make more robust
        self.out_sz //= stride
        self.output = activation_by_name(actvn)(conv_out +
                                         self.b.dimshuffle('x', 0, 'x', 'x'))

        # Store Parameters
        self.params = [self.W, self.b]
        self.inpt = inpt
        self.num_maps = num_maps
        self.mode = mode
        self.n_out = num_maps * self.out_sz ** 2
        self.reg = {"L1": 0, "L2": 0,
                    "momentum": .95,
                    "rate": 1,
                    "maxnorm": 0,}
        self.reg.update(reg)

        self.args = (batch_sz, num_prev_maps, in_sz, num_maps, filter_sz,
                     stride, mode, actvn, reg)
        self.representation = (
            "Conv Maps:{:2d} Filter:{} Stride:{} Mode:{} Output:{:2d} "
            "Act:{}\n\t  L1:{L1} L2:{L2} Momentum:{momentum} Rate:{rate} Max Norm:{maxnorm}"
            "".format(num_maps, filter_sz, stride, mode, self.out_sz,
                      actvn, **self.reg))

    def TestVersion(self, inpt):
        return ConvLayer(inpt, (self.W, self.b), None, *self.args)

class PoolLayer(Layer):
    def __init__(self, inpt, num_maps, in_sz, pool_sz, ignore_border=False):
        """
        Pool Layer to follow Convolutional Layer
        :param inpt:
        :param pool_sz:
        :param ignore_border: When True, (5,5) input with ds=(2,2)
            will generate a (2,2) output. (3,3) otherwise.
        """
        self.output = downsample.max_pool_2d(inpt, (pool_sz, pool_sz),
                                             ignore_border=ignore_border)

        if ignore_border:
            self.out_sz = in_sz//pool_sz
        else:
            self.out_sz = math.ceil(in_sz/pool_sz)

        self.params = []
        self.inpt = inpt
        self.num_maps = num_maps
        self.ignore_border = ignore_border
        self.args = (num_maps, in_sz, pool_sz, ignore_border)
        self.n_out = num_maps * self.out_sz ** 2
        self.representation = (
            "Pool Maps:{:2d} Pool_sz:{} Border:{} Output:{:2d}"
            "".format(num_maps, pool_sz,
                      "Ignore" if ignore_border else "Keep",
                      self.out_sz))

    def TestVersion(self, inpt):
        return PoolLayer(inpt, *self.args)

class MeanLayer(Layer):
    def __init__(self, inpt, num_maps, in_sz):
        self.output = tt.mean(inpt, axis=(2,3))
        self.params = []
        self.inpt = inpt
        self.num_maps = num_maps
        self.in_sz = in_sz
        self.out_sz = 1
        self.n_out = num_maps

        self.representation = (
            "Mean Maps:{:2d} Output:{:2d}"
            "".format(num_maps, self.out_sz))

    def TestVersion(self, inpt):
        return MeanLayer(inpt, self.num_maps, self.in_sz)
