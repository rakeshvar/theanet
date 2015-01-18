import numpy as np
import theano as th
import theano.tensor as tt
from hidden import HiddenLayer
from layer import Layer, activation_by_name
from outlayers import OutputLayer

from weights import init_wb, borrow

float_x = th.config.floatX

############################## Location 'Layer' ###############################


class LocationInfo():
    def __init__(self,
                 wts,
                 rand_gen=None,
                 n_aux=(6, 6),
                 test_version=False,):

        loc_inpt4 = tt.tensor3('AuxiallaryInput')

        if not test_version:
            srs = tt.shared_randomstreams.RandomStreams(rand_gen.randint(1e6)
                                                        if rand_gen else None)
            u = srs.uniform(size=(loc_inpt4.shape[0],)).dimshuffle(0, 'x')
            loc_inpt2 = loc_inpt4[:, 0, :] * u + loc_inpt4[:, 1, :] * (1 - u)

        else:
            loc_inpt2 = tt.mean(loc_inpt4, axis=1)

        n_aux_hid, n_aux_out = n_aux

        # First Layer
        actvn1 = "softplus"
        loc1_wts = None if wts is None else wts[:2]
        loc1_w, loc1_b = init_wb(loc1_wts, rand_gen,
                                 (2, n_aux_hid), n_aux_hid,
                                 n_aux_hid + 2, n_aux_hid + 2,
                                 actvn1, 'Loc1')
        hidden = activation_by_name(actvn1)(tt.dot(loc_inpt2, loc1_w) +
                                              loc1_b)

        # Second Layer
        actvn2 = "linear"
        loc2_wts = None if wts is None else wts[2:]
        loc2_w, loc2_b = init_wb(loc2_wts, rand_gen,
                                 (n_aux_hid, n_aux_out), n_aux_out,
                                 n_aux_out + n_aux_hid, n_aux_out + n_aux_hid,
                                 actvn2, 'Loc2')
        self.output = activation_by_name(actvn2)(tt.dot(hidden, loc2_w) +
                                                 loc2_b)

        self.aux_inpt = loc_inpt4
        self.params = [loc1_w, loc1_b, loc2_w, loc2_b]


############################## Auxiallary Layers ###############################


class AuxConcatLayer(Layer):
    def __init__(self,
                 inpt,
                 wts,
                 rand_gen,
                 n_in,
                 n_aux,
                 aux_type,
                 test_version=False):
        """
        """
        # noinspection PyCallingNonCallable
        aux_info = globals()[aux_type](wts, rand_gen, n_aux, test_version)
        output = tt.concatenate((inpt, aux_info.output), axis=1)

        self.aux_inpt = aux_info.aux_inpt
        self.output = output
        self.n_aux = n_aux
        self.n_in = n_in
        self.n_out = n_aux + n_in
        self.aux_type = aux_type
        self.params = aux_info.params
        self.representation = "AuxConcat In:{:3d} Aux:{} Out:{:3d} ". \
            format(n_in, n_aux, self.n_out)

    def TestVersion(self, te_inpt):
        return AuxConcatLayer(te_inpt,
                              self.params, None,
                              self.n_in, self.n_aux,
                              self.aux_type,
                              test_version=True)


class SoftAuxLayer(HiddenLayer, OutputLayer):
    def __init__(self,
                 inpt,
                 wts,
                 rand_gen,
                 n_in, n_out, n_aux,
                 aux_type,
                 test_version=False):

        hidden_wts = None if wts is None else wts[:2]
        HiddenLayer.__init__(self, inpt, hidden_wts, rand_gen, n_in, n_out,
                             actvn='linear',
                             pdrop=0)

        aux_wts = None if wts is None else wts[2:6]
        # noinspection PyCallingNonCallable
        aux_info = globals()[aux_type](aux_wts, rand_gen, n_aux, test_version)

        cross_wts = None if wts is None else wts[6:]
        n_aux_hid, n_aux_out = n_aux
        cross_w, cross_b = init_wb(cross_wts, rand_gen,
                                   (n_aux_out, n_out), n_out,
                                   n_aux_out + n_out, n_aux_out + n_out,
                                   'relu', 'SoftAuxCross')

        self.hidden_ouput = self.output #* 0.0 + cross_b
        self.output = tt.nnet.softmax(self.hidden_ouput +
                                      tt.dot(aux_info.output, cross_w))

        self.aux_inpt = aux_info.aux_inpt
        self.n_aux = n_aux
        self.n_out = n_out
        self.aux_type = aux_type
        self.params += aux_info.params
        self.params += [cross_w, cross_b]
        self.representation = "SoftAux In:{:3d} Aux:{} Out:{:3d}" \
                              "".format(n_in, n_aux, n_out,)

        #############################################################
        self.y_preds = tt.argmax(self.output, axis=1)
        self.probs = self.output
        self.logprob = tt.log(self.probs)
        self.features = self.logprob
        self.kind = 'SOFTMAX'

    def TestVersion(self, inpt):
        return SoftAuxLayer(inpt, self.params, rand_gen=None,
                            n_in=self.n_in, n_out=self.n_out, n_aux=self.n_aux,
                            aux_type=self.aux_type,
                            test_version=True)