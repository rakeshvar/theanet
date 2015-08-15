import numpy as np
import pickle
import sys
import pprint

def wts_info(d, gory=False):
    ret = ""
    tot_wts = 0

    for i, wb in enumerate(d['allwts']):
        ret += "\nLayer : {}".format(i)
        if len(wb) == 0:
            continue

        for w in wb:
            rms = (w**2).mean()**.5
            ret += "\n\t{}".format(w.shape)
            ret += "\n\tMin: {:+.2f} Avg: {:.2f} Max: {:+.2f}".format(
                w.min(), w.mean(), w.max())
            p = np.prod(w.shape[1:])
            tot_wts += np.prod(w.shape) 

            if p > 1:
                ret += "\n\tnin: {:.0f}(1/{:4.3f}) rms:{:5.2f}({:.2f})" \
                       "".format(p, 1/p, rms, rms*np.sqrt(p))
                if w.ndim == 2:
                    norms = (w**2).sum(axis=0)**.5
                    ret += "\n\tNorms:{:.2f} {:.2f} {:.2f}".format(
                        norms.min(), norms.mean(), norms.max())

            if gory:
                ret += "\n" + "-"*80 + "\n"
                ret += str(w)

    ret += "\nTotal number of weights: {:.0f}".format(tot_wts)
    return ret


for pkl_fname in sys.argv[1:]:
    with open(pkl_fname, 'rb') as f:
        d = pickle.load(f)

    print(pkl_fname)
    pprint.PrettyPrinter(indent=4).pprint(d["layers"])
    #print(wts_info(d, gory=True))
    print(wts_info(d))
