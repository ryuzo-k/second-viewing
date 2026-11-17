"""Loader for DANDI 001187 (encoding phase of working-memory trials, then a recognition test). Reads encoding-phase
spikes and the test-phase table (image, answer, confidence, times); test-phase spikes are never read here."""
import csv, functools, glob, os
import numpy as np, h5py
import analysis as A

IMAGE_SET = {r['file']: int(r['image_set']) for r in csv.DictReader(open(os.path.join(A.INPUTS, 'image_sets.csv')))}


def cat_of(i):
    """Category (1-5) of a 001187 image ID; 0 for images outside the five categories."""
    if i < 900: return i // 100
    if 1001 <= i <= 1200: return (i - 1001) // 40 + 1
    return 0


def files():
    if not A.DATA: raise SystemExit('Set DATA to the folder holding 001187/ and 000004/.')
    return sorted(p for p in glob.glob(os.path.join(A.DATA, '001187', '*', '*.nwb')) if not os.path.basename(p).startswith('._'))


@functools.lru_cache(None)
def typicality():
    """(image set, image ID) -> typicality of each test image (made by typicality.py)."""
    return {(int(r['image_set']), int(r['image_id'])): float(r['typicality'])
            for r in csv.DictReader(open(os.path.join(A.INPUTS, 'typicality.csv')))}


def load_session(path):
    """Encoding-phase spike counts per presentation and the test-phase table."""
    f = h5py.File(path, 'r'); W = f['intervals/WM_trials']; L = f['intervals/LTM_trials']
    fn = os.path.basename(path); v = IMAGE_SET[fn]
    load = W['loads'][:]; wm_end = W['stop_time'][-1]
    enc_on, enc_id, h_on, enc_ld = [], [], [], []
    tm_ = W['timestamps_Maintenance'][:]
    for k in (1, 2, 3):
        m = np.ones(len(load), bool) if k == 1 else (load == 3)
        enc_on.append(W[f'timestamps_Encoding{k}'][:][m]); enc_id.append(W[f'PicIDs_Encoding{k}'][:][m]); enc_ld.append(load[m])
        # a hold period follows this image only if it is the last one of its trial (load 1: image 1; load 3: image 3)
        last = (load[m] == 1) if k == 1 else (np.full(m.sum(), k == 3))
        h_on.append(np.where(last, tm_[m], np.nan))
    enc_k = np.concatenate([np.full(len(x), k) for k, x in zip((1, 2, 3), enc_on)])   # 1st/2nd/3rd image of its trial
    enc_on = np.concatenate(enc_on); enc_id = np.concatenate(enc_id).astype(int); h_on = np.concatenate(h_on)
    enc_ld = np.concatenate(enc_ld)
    enc_t = np.argsort(np.argsort(enc_on)) / max(len(enc_on) - 1, 1) - 0.5      # time in the encoding phase (rank, -0.5 .. 0.5)
    st = f['units/spike_times'][:]; idx = f['units/spike_times_index'][:].astype(np.int64)
    starts = np.concatenate([[0], idx[:-1]]).astype(np.int64); el = f['units/electrodes'][:]
    eid = f['general/extracellular_ephys/electrodes/id'][:]
    loc = dict(zip(eid, [x.decode() if isinstance(x, bytes) else str(x) for x in f['general/extracellular_ephys/electrodes/location'][:]]))
    enc = np.zeros((len(idx), len(enc_on))); hold = np.full((len(idx), len(enc_on)), np.nan); trains = []
    pre = np.zeros((len(idx), len(enc_on))); enc_s = np.zeros((len(idx), len(enc_on)))
    hm = ~np.isnan(h_on); ho = h_on[hm]
    for u in range(len(idx)):
        sp = np.sort(st[starts[u]:idx[u]]); sp = sp[sp <= wm_end + 1]; trains.append(sp)
        enc[u] = np.searchsorted(sp, enc_on + A.X_WINDOW[1]) - np.searchsorted(sp, enc_on + A.X_WINDOW[0])
        enc_s[u] = np.searchsorted(sp, enc_on + A.TEST_WINDOW[1]) - np.searchsorted(sp, enc_on + A.TEST_WINDOW[0])
        hold[u, hm] = (np.searchsorted(sp, ho + A.HOLD[1]) - np.searchsorted(sp, ho + A.HOLD[0])) * (A.X_WINDOW[1] - A.X_WINDOW[0]) / (A.HOLD[1] - A.HOLD[0])
        pre[u] = (np.searchsorted(sp, enc_on + A.PRE[1]) - np.searchsorted(sp, enc_on + A.PRE[0])) * (A.X_WINDOW[1] - A.X_WINDOW[0]) / (A.PRE[1] - A.PRE[0])
    on = L['timestamps_PicOnset'][:]; rt = L['timestamps_Response'][:] - on
    conf = L['confidence'][:].astype(int); ids = L['PicIDs'][:].astype(int); shown = L['num_times_shown'][:].astype(int)
    n = len(on); typ = typicality()
    test = dict(id=ids, shown=shown, said_old=(conf >= 4).astype(int), conf=conf, rt=rt,
                order=np.arange(n) / max(n - 1, 1) - 0.5,
                prev_old=np.r_[0, (shown[:-1] > 0).astype(int)], prev_said=np.r_[0, (conf[:-1] >= 4).astype(int)],
                typicality=np.array([typ[(v, i)] for i in ids]))   # every test image has a value
    test['keep_pairs'] = (rt >= A.MIN_RT) & (conf >= 1) & (conf <= 6) & np.isin(shown, [0, 1]) & np.array([cat_of(i) > 0 for i in ids])
    test['keep_repetition'] = (rt >= A.MIN_RT) & (conf >= 1) & (conf <= 6) & np.isin(shown, [0, 1, 2]) & np.array([cat_of(i) > 0 for i in ids])
    last = {}
    for i_, t_ in zip(enc_id, enc_on): last[i_] = max(last.get(i_, -np.inf), t_)
    for i_, t_ in zip(W['PicIDs_Probe'][:].astype(int), W['timestamps_Probe'][:]): last[i_] = max(last.get(i_, -np.inf), t_)
    test['delay'] = np.array([np.log(max(o - last[i_], 1.0)) if i_ in last else np.nan for i_, o in zip(ids, on)])
    pk = dict(zip(W['PicIDs_Probe'][:].astype(int), W['probe_in_out'][:]))
    test['probe_kind'] = np.array([(1 if pk[i_] == 1 else 2) if i_ in pk else 0 for i_ in ids])   # 1 same trial, 2 earlier trial
    never_tested = ~np.isin(enc_id, ids)    # encoding-phase images never tested: used to select neurons and to measure x
    sess = dict(file=fn, participant=path.split('/')[-2], image_set=v, enc=enc, enc_id=enc_id, never_tested=never_tested,
                trains=trains, hold=hold, has_hold=hm, pre=pre, enc_k=enc_k, enc_s=enc_s, enc_on=enc_on, enc_load=enc_ld,
                enc_t=enc_t, enc_cat=np.array([cat_of(i) for i in enc_id]), elec=el, area=[loc.get(e, '') for e in el], test=test)
    A.set_pop(sess)
    return sess
