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
    def __init__(self, wts, rand_gen=None,
                 n_aux=6, actvn='tanh', test_version=False):
        loc_inpt4 = tt.tensor3('AuxiallaryInput')

        if not test_version:
            srs = tt.shared_randomstreams.RandomStreams(rand_gen.randint(1e6)
                                                        if rand_gen else None)
            u = srs.uniform(size=(loc_inpt4.shape[0],)).dimshuffle(0, 'x')
            loc_inpt2 = loc_inpt4[:, 0, :] * u + loc_inpt4[:, 1, :] * (1 - u)

        else:
            loc_inpt2 = tt.mean(loc_inpt4, axis=1)

        hidden_w = th.shared(np.asarray([[1, 0, -1, 0, 1, 1],
                                         [0, 1, 0, -1, 1, -1]],
                                        dtype=th.config.floatX))
        n_hidden = borrow(hidden_w).shape[1]
        hidden = tt.maximum(tt.dot(loc_inpt2, hidden_w), 0)

        loc_w, loc_b = init_wb(wts, rand_gen, (n_hidden, n_aux), n_aux,
                               n_aux + n_hidden, n_aux + n_hidden, 'relu',
                               'Location')
        self.output = activation_by_name(actvn)(tt.dot(hidden, loc_w) + loc_b)
        self.aux_inpt = loc_inpt4
        self.params = [loc_w, loc_b]


############################## Auxiallary Layers ###############################


class AuxConcatLayer(Layer):
    def __init__(self,
                 inpt,
                 wts,
                 rand_gen,
                 n_in,
                 n_aux,
                 aux_type,
                 aux_actvn,
                 test_version=False):
        """
        """
        # noinspection PyCallingNonCallable
        aux_info = globals()[aux_type](wts, rand_gen, n_aux, aux_actvn,
                                       test_version)
        output = tt.concatenate((inpt, aux_info.output), axis=1)

        self.aux_inpt = aux_info.aux_inpt
        self.output = output
        self.n_aux, self.n_in = n_aux, n_in
        self.n_out = n_aux + n_in
        self.aux_type = aux_type
        self.aux_actvn = aux_actvn
        self.params = aux_info.params
        self.representation = "AuxConcat In:{:3d} Aux:{:3d} Out:{:3d} Act:{}". \
            format(n_in, n_aux, self.n_out, aux_actvn, )

    def TestVersion(self, te_inpt):
        return AuxConcatLayer(te_inpt,
                              self.params, None,
                              self.n_in, self.n_aux,
                              self.aux_type,
                              self.aux_actvn,
                              test_version=True)


class SoftAuxLayer(HiddenLayer, OutputLayer):
    def __init__(self,
                 inpt,
                 wts,
                 rand_gen,
                 n_in, n_out, n_aux,
                 aux_type,
                 aux_actvn,
                 test_version=False):

        hidden_wts = None if wts is None else wts[:2]
        HiddenLayer.__init__(self, inpt, hidden_wts, rand_gen, n_in, n_out,
                             actvn='linear',
                             pdrop=0)

        aux_wts = None if wts is None else wts[2:4]
        # noinspection PyCallingNonCallable
        aux_info = globals()[aux_type](aux_wts, rand_gen, n_aux, aux_actvn,
                                       test_version)

        oth_wts = None if wts is None else wts[4:6]
        oth_w, oth_b = init_wb(oth_wts, rand_gen, (n_aux, n_out), n_out,
                               n_aux + n_out, n_aux + n_out, 'relu',
                               'SoftAuxCross')

        self.hidden_ouput = self.output
        self.output = tt.nnet.softmax(self.hidden_ouput +
                                      tt.dot(aux_info.output, oth_w))

        self.aux_inpt = aux_info.aux_inpt
        self.n_aux = n_aux
        self.n_out = n_out
        self.aux_type = aux_type
        self.aux_actvn = aux_actvn
        self.params += aux_info.params
        self.params += [oth_w, oth_b]
        self.representation = "SoftAux In:{:3d} Aux:{:3d} Out:{:3d} AuxAct:{}" \
                              "".format(n_in, n_aux, n_out, aux_actvn, )

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
                            aux_actvn=self.aux_actvn,
                            test_version=True)