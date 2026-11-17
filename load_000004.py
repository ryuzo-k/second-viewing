"""Loader for DANDI 000004 (learning phase, then a recognition test). Reads learning-phase spikes and the recognition
table (image, old/new, confidence, times); recognition-phase spikes are never read here. The learning phase plays the
role of the 001187 encoding phase, the recognition phase that of the test phase."""
import glob, os
import numpy as np, h5py
import analysis as A

PIC = {}; CAT = {}; LISTS = {}     # image file -> ID; ID -> category; (learned, tested) image sets -> list number


def _str(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def files():
    if not A.DATA: raise SystemExit('Set DATA to the folder holding 001187/ and 000004/.')
    return sorted(p for p in glob.glob(os.path.join(A.DATA, '000004', '*', '*.nwb')) if not os.path.basename(p).startswith('._'))


def _image_id(name, cat):
    if name not in PIC: PIC[name] = len(PIC) + 1; CAT[PIC[name]] = cat
    return PIC[name]


def load_session(path):
    """Session in the same layout as load_001187.load_session. Each participant's image list (the same images with the
    same old/new status) plays the role of the 001187 image set."""
    f = h5py.File(path, 'r'); t = f['intervals/trials']
    ph = np.array([_str(x) for x in t['stim_phase'][:]]); L = ph == 'learn'; Rg = ph == 'recog'
    names = [_str(x).replace('\\', '/') for x in t['external_image_file'][:]]; cats = t['stimCategory'][:].astype(int)
    ids = np.array([_image_id(n, c) for n, c in zip(names, cats)])
    on = t['stim_on_time'][:]
    st = f['units/spike_times'][:]; idx = f['units/spike_times_index'][:].astype(np.int64)
    starts = np.concatenate([[0], idx[:-1]]).astype(np.int64); el = f['units/electrodes'][:]
    eid = f['general/extracellular_ephys/electrodes/id'][:]
    loc = dict(zip(eid, [_str(x) for x in f['general/extracellular_ephys/electrodes/location'][:]]))
    learn_end = on[L].max() + 3
    enc_on = on[L]; enc = np.zeros((len(idx), L.sum())); trains = []; pre = np.zeros((len(idx), L.sum()))
    for u in range(len(idx)):
        sp = np.sort(st[starts[u]:idx[u]]); sp = sp[sp <= learn_end]; trains.append(sp)
        enc[u] = np.searchsorted(sp, enc_on + A.X_WINDOW[1]) - np.searchsorted(sp, enc_on + A.X_WINDOW[0])
        pre[u] = (np.searchsorted(sp, enc_on + A.PRE[1]) - np.searchsorted(sp, enc_on + A.PRE[0])) * (A.X_WINDOW[1] - A.X_WINDOW[0]) / (A.PRE[1] - A.PRE[0])
    tid = ids[Rg]; learned = set(ids[L])
    conf = (t['response_value'][:][Rg] - 30).astype(int)
    rt = t['response_time'][:][Rg] - on[Rg]
    shown = np.array([int(i in learned) for i in tid]); n = Rg.sum()
    test = dict(id=tid, shown=shown, said_old=(conf >= 4).astype(int), conf=conf, rt=rt,
                order=np.arange(n) / max(n - 1, 1) - 0.5,
                prev_old=np.r_[0, shown[:-1]], prev_said=np.r_[0, (conf[:-1] >= 4).astype(int)], typicality=np.zeros(n))
    test['keep_pairs'] = (rt >= A.MIN_RT) & (conf >= 1) & (conf <= 6)
    enc_id = ids[L]; never_tested = ~np.isin(enc_id, tid)
    key = (frozenset(enc_id.tolist()), frozenset(tid.tolist()))
    lst = LISTS.setdefault(key, len(LISTS) + 1)
    sess = dict(file=os.path.basename(path), participant=path.split('/')[-2], image_set=lst, enc=enc, enc_id=enc_id, never_tested=never_tested, trains=trains,
                enc_cat=cats[L], elec=el, area=[loc.get(e, '') for e in el], test=test,
                pre=pre, enc_k=np.ones(L.sum(), int))                      # each learning image is its own trial
    A.set_pop(sess)
    return sess
