import numpy as np
import pickle
import sys
import pprint

def wts_info(wb):
    ret, tot_wts = "", 0
    for w in wb:
        n_wts = np.prod(w.shape)
        tot_wts += n_wts
        n_in = np.prod(w.shape[1:])

        ret += "\n    " + "WB"[n_in==1]
        ret += "\n\tShape:{} = {:,}".format(w.shape, n_wts)
        ret += "\n\tMin={:+.2f} Avg={:.2f} Max={:+.2f}".format(
            w.min(), w.mean(), w.max())

        if n_in > 1:
            rms = (w**2).mean()**.5
            ret += "\n\tnin={:.0f}" \
                   "\n\trms={:5.2f} (√nin rms={:.2f})" \
                   "".format(n_in, rms, rms*np.sqrt(n_in))
            sum_along = 0 if w.ndim == 2 else (1, 2, 3)
            norms = (w**2).sum(axis=sum_along)**.5
            ret += "\n\tNorms:{:.2f} {:.2f} {:.2f}".format(
                norms.min(), norms.mean(), norms.max())

    return ret, tot_wts

def all_info(d):
    tot_wts = 0
    for i, (layer, wb) in enumerate(zip(d["layers"], d['allwts'])):
        print("{:2d} {} \n   Params".format(i, layer[0]))
        for k in sorted(layer[1].keys()):
            print("\t'{}': {}".format(k, layer[1][k]))
        info, nwts = wts_info(wb)
        print(info)
        tot_wts += nwts

    print("\nTotal Number of Weights: {:,}".format(tot_wts))


for pkl_fname in sys.argv[1:]:
    with open(pkl_fname, 'rb') as f:
        d = pickle.load(f)

    print(pkl_fname)
    all_info(d)
