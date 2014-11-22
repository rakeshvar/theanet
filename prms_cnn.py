layers = [
        ('InputLayer', {
            'max_perturb'   :4,
            'img_sz'        :68,
            'pflip'         :0,
            'batch_sz'      :20,
            'num_maps'      :1,
            }),
        ('ConvPoolLayer', {
            'num_maps'      :6,
            'filter_sz'     :5,
            'stride'        :1,
            'pool_sz'       :2,
            }),
        ('ConvPoolLayer', {
            'num_maps'      :60,
            'filter_sz'     :2,
            'stride'        :1,
            'pool_sz'       :2,
            'actvn'         :"relu",
            }),
        ('HiddenLayer', {
            'n_out'         :1000,
            'pdrop'         :.5,
            'actvn'         :'relu',
            }),
        # ('HiddenLayer', {
        #     'n_out'         :2000,
        #     'pdrop'         :.5,
        #     'actvn'         :'relu',
        #     }),
#       ('CenteredOutLayer', {
#           'centers'       :None,
#           'n_features'    :100,
#           'n_classes'     :10,
#           'kind'          :'RBF',
#           'learn_centers' :True, 
#           'junk_dist'     :1e6,
#           }),
        ('SoftmaxLayer', {
            'n_out'         :460,
            }),
]

training_params = {
    'NUM_EPOCHS' : 21,
    'TRAIN_ON_FRACTION' : .75,
    'EPOCHS_TO_TEST' : 4,
    'TEST_SAMP_SZ': 5000,
    'DEFORM'    : 'parallel',
    'DFM_PRMS' : {'scale' : 64, 'sigma' : 8, 'cval'  : 1, 'ncpus' : 4},

    'MOMENTUM' : .95,
    'INIT_LEARNING_RATE': .1,
    'EPOCHS_TO_HALF_RATE':  1,
    'LAMBDA1': 0.0,
    'LAMBDA2': 0.001,
    'MAXNORM': 4,
}

# TODO:
    # Move regularization to layer
