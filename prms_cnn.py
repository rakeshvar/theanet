layers = [
        ('DeformLayer', {
            'translation'   :4,
            'img_sz'        :68,
            'zoom'          :1.1,
            'magnitude'     :24,
            'sigma'         :8,
            'pflip'         :1./64,
            }),
        ('ConvPoolLayer', {
            'num_maps'      :6,
            'filter_sz'     :5,
            'stride'        :1,
            'pool_sz'       :2,
            }),
        ('ConvPoolLayer', {
            'num_maps'      :60,
            'filter_sz'     :3,
            'stride'        :1,
            'pool_sz'       :2,
            'actvn'         :"relu",
            }),
        ('HiddenLayer', {
            'n_out'         :1000,
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
    'BATCH_SZ'   :20,
    'NUM_EPOCHS' : 61,
    'TRAIN_ON_FRACTION' : .75,
    'EPOCHS_TO_TEST' : 4,
    'TEST_SAMP_SZ': 5000,
    'DEFORM'    : 'none',
    'DFM_PRMS' : {},

    'MOMENTUM' : .95,
    'INIT_LEARNING_RATE': .1,
    'EPOCHS_TO_HALF_RATE':  1,
    'LAMBDA1': 0.0,
    'LAMBDA2': 0.001,
    'MAXNORM': 4,
}

# TODO:
    # Move regularization to layer
