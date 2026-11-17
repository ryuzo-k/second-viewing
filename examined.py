"""Encoding-phase numbers disclosed in the preregistration, computed from encoding-phase spikes and test-phase
behaviour only; no test-phase spike is read.
  first_viewing     at first viewing (TEST_WINDOW), twice- vs once-seen images x x: the first-viewing part of the
                    repetition model, without and with the first-viewing context terms; SD over 40 resamples of
                    participants
  pilot_pairs       the pair analysis on encoding-phase probes, which re-show an image from the same trial (seconds
                    earlier) or from an earlier trial: the probe response plays the test response, the in/out answer
                    the old/new answer; in the form fixed before the pilot, without the test response of the other
                    neurons as a covariate (added to the pair analysis afterwards)
  x                 quartiles of x over tuned neurons x categories
  delays            time from the last viewing to the test (old test images) and the gap between the phases, minutes

Usage: DATA=... python examined.py"""
import json
import numpy as np, h5py
import analysis as A
import load_001187 as L


def sessions():
    A.category = L.cat_of; out = []
    for p in L.files():
        s = L.load_session(p); s['tuned'] = A.tuned_neurons(s); s['path'] = p; out.append(s)
    return out


def first_viewing(S, n=40, seed=0):
    rows = []; code = {}
    for j, s in enumerate(S):
        T = s['test']; fi = A._first_index(s); pj = code.setdefault(s['participant'], len(code))
        for u in s['tuned']:
            xs = A.usual_response(s, u)
            for t in np.where(T['keep_repetition'] & (T['shown'] >= 1))[0]:
                i = T['id'][t]
                if i not in fi: continue
                f = fi[i]; c = L.cat_of(i)
                rows.append((1000000 * j + u * 100 + c * 10, s['enc_s'][u, f], float(T['shown'][t] == 2), xs[c],
                             s['enc_t'][f], float(s['enc_load'][f] == 1), float(s['enc_k'][f] == 1), pj))
    R = np.array(rows, float)

    def fit(w, context):
        m = w > 0; Q = R[m]; w = w[m]
        g, y, tw, x = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]; cat = (np.round(g) // 10) % 10
        cols = [tw, tw * x, np.column_stack([tw * (cat == c) for c in range(2, 6)])]
        if context: ctx = Q[:, 4:7] - Q[:, 4:7].mean(0); cols += [ctx, ctx * x[:, None]]
        return A.pois_fe(g, y, np.column_stack(cols), np.zeros(len(Q)), w)[1]
    part = R[:, 7].astype(int); npart = part.max() + 1; rng = np.random.default_rng(seed)
    W = [np.bincount(rng.integers(0, npart, npart), minlength=npart)[part].astype(float) for _ in range(n)]
    return {name: dict(k_first_viewing=round(float(fit(np.ones(len(R)), ctx)), 4), sd=round(float(np.std([fit(w, ctx) for w in W])), 4))
            for name, ctx in (('plain', False), ('with_context', True))}


def probe_session(s, same_trial):
    """s with its test table replaced by the encoding-phase probes of one kind and their spike counts."""
    f = h5py.File(s['path'], 'r'); W = f['intervals/WM_trials']
    pid = W['PicIDs_Probe'][:].astype(int); tp = W['timestamps_Probe'][:]; pin = W['probe_in_out'][:]
    rt = W['timestamps_Response'][:] - tp; said_in = np.where(W['response_accuracy'][:] == 1, pin, 1 - pin)
    sel = (pin == (1 if same_trial else 0)) & np.array([L.cat_of(i) > 0 for i in pid])
    pid, tp, rt, said_in = pid[sel], tp[sel], rt[sel], said_in[sel]
    first_t = {}
    for k in (1, 2, 3):
        ids = W[f'PicIDs_Encoding{k}'][:].astype(int); ts = W[f'timestamps_Encoding{k}'][:]; ld = W['loads'][:]
        for i, t, l in zip(ids, ts, ld):
            if (k == 1 or l == 3) and i > 0: first_t[i] = min(first_t.get(i, np.inf), t)
    before = np.array([first_t.get(i, np.inf) < t for i, t in zip(pid, tp)])    # the first showing precedes the probe
    n = len(pid); conf = np.where(said_in == 1, 6, 1)
    c = dict(s); c['test'] = dict(id=pid, shown=before.astype(int), said_old=(conf >= 4).astype(int), conf=conf, rt=rt,
                                   keep_pairs=(rt >= A.MIN_RT) & before)
    st = f['units/spike_times'][:]; idx = f['units/spike_times_index'][:].astype(np.int64); starts = np.r_[0, idx[:-1]]
    tc = np.zeros((len(idx), n))
    for u in range(len(idx)):
        sp = np.sort(st[starts[u]:idx[u]])
        tc[u] = np.searchsorted(sp, tp + A.TEST_WINDOW[1]) - np.searchsorted(sp, tp + A.TEST_WINDOW[0])
    c['test_count'] = tc
    return c


def pilot_pairs(S):
    out = {}
    for name, same in (('earlier_trial', False), ('same_trial', True)):
        res = A.pairs([probe_session(s, same) for s in S], phase='enc', test_population=False)
        pv = A.pair_swap_test(res, np.random.default_rng(20261119), n=1000)
        out[name] = dict(z=round(pv['z'], 2), best=round(pv['best'], 4), n_B=sum(len(v['res']) for v in res.values()))
    return out


def x_distribution(S):
    x = []
    for s in S:
        for u in s['tuned']:
            e = s['enc'][u]; m = e[s['never_tested']].mean()
            x += [e[s['never_tested'] & (s['enc_cat'] == c)].mean() / m - 1 for c in range(1, 6)]
    return dict(tuned_neurons=int(sum(len(s['tuned']) for s in S)), quartiles=np.round(np.percentile(x, [25, 50, 75]), 3).tolist())


def delays(S):
    d = np.concatenate([np.exp(s['test']['delay'][(s['test']['shown'] >= 1) & np.isfinite(s['test']['delay'])]) for s in S]) / 60
    gap = []
    for s in S:
        f = h5py.File(s['path'], 'r'); gap.append((f['intervals/LTM_trials']['start_time'][0] - f['intervals/WM_trials']['stop_time'][-1]) / 60)
    return dict(last_viewing_to_test_median=round(float(np.median(d)), 1), last_viewing_to_test_5_95=np.round(np.percentile(d, [5, 95]), 1).tolist(),
                gap_between_phases_median=round(float(np.median(gap)), 1), gap_between_phases_range=[round(float(min(gap)), 1), round(float(max(gap)), 1)])


if __name__ == '__main__':
    S = sessions()
    print(json.dumps(dict(x=x_distribution(S), first_viewing=first_viewing(S), delays=delays(S), pilot_pairs=pilot_pairs(S)), indent=1))
