layers = [
		('InputLayer', {
			'max_perturb'	:0,
			'img_sz'		:32,
			'pflip'			:.1,
			'batch_sz'		:20,
			'num_maps'		:1,
			}),
		('ConvPoolLayer', {
			'num_maps' 		:6,
			'filter_sz'		:5,
			'stride'		:1,
			'pool_sz'		:2,
			'actvn'			:"tanh",
			}),
		('HiddenLayer', {
			'n_out'			:60,
			'pdrop'			:.5,
			}),
		('SoftmaxLayer', {
			'n_out'			:10,
			}),
]

training_params = {
	'NUM_EPOCHS' : 21,
	'TRAIN_ON_FRACTION' : .75,
	'EPOCHS_TO_TEST' : 4,
	'TEST_SAMP_SZ': 1000,
    'DEFORM'    : 'parallel',
    'DFM_PRMS' : {'scale' : 32, 'sigma' : 4, 'cval'  : 0,},

	'MOMENTUM' : .9,
	'SEED'	: None,
	'INIT_LEARNING_RATE': .1,
	'EPOCHS_TO_HALF_RATE':	12,
	'LAMBDA1': 0.,
	'LAMBDA2': 0.0001,
	'MAXNORM': 3.5,
}

# TODO:
	# Move regularization to layer
