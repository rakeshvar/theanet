layers = [
        ('InputLayer', {
            'max_perturb'   :4,
            'img_sz'        :68,
            'pflip'         :.05,
            'batch_sz'      :20,
            'num_maps'      :1,
            }),
        ('HiddenLayer', {
            'n_out'         :4096,
            'pdrop'         :.5,
            'actvn'         :'relu',
            }),
        ('HiddenLayer', {
            'n_out'         :4096,
            'pdrop'         :.5,
            'actvn'         :'relu',
            }),
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
    'NUM_EPOCHS' : 200,
    'TRAIN_ON_FRACTION' : .75,
    'EPOCHS_TO_TEST' : 4,
    'TEST_SAMP_SZ': 5000,
    'DEFORM'    : 'parallel',
    'DFM_PRMS' : {'scale' : 64, 'sigma' : 8, 'cval'  : 0,},

    'MOMENTUM' : .9,
    'INIT_LEARNING_RATE': .3,
    'EPOCHS_TO_HALF_RATE':  8,
    'LAMBDA1': 0.001,
    'LAMBDA2': 0.0,
    'MAXNORM': 3.5,
}

# TODO:
    # Move regularization to layer
