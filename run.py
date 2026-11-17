"""Locked analysis of the study.

python run.py          DANDI 001187. Excludes unstable neurons, then runs the preregistered test (repetition) and writes
                       it to results/001187_repetition.json before anything else; a NaN stops the run. Then the
                       exploratory pair analysis with its controls, the descriptive checks and the counts
                       (results/001187.json); an error there is recorded, never used.
python run.py 000004   DANDI 000004, pair analysis only. Runs only after POSTED_001187_SHA.txt holds the SHA-256 of
                       results/001187.json; a test only if the 001187 pair analysis and its controls passed.

Test-phase spikes are read only when UNBLIND=after-registration and OSF_REGISTRATION_URL.txt holds the registration.
DRYRUN=1 runs everything on synthetic test counts; no test-phase spike is read."""
import hashlib, json, os, sys
import numpy as np, h5py
import analysis as A
import load_001187
from simulate import simulate_session

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, 'OSF_REGISTRATION_URL.txt')
POSTED = os.path.join(HERE, 'POSTED_001187_SHA.txt')
RES = os.path.join(HERE, 'results')
DRY = os.environ.get('DRYRUN') == '1'
SEED_1187 = 20261117; SEED_0004 = 20261118; NDRAW = 1000
# neuron stability, encoding vs test phase: 1st/99th percentiles of the same quantities within the encoding phase;
# share of inter-spike intervals < 3 ms
WF_CORR, AMP_LO, AMP_HI, RATE_LO, RATE_HI, ISI_MAX = 0.98, 0.86, 1.12, 0.33, 3.3, 0.01


def guard():
    if os.environ.get('UNBLIND') != 'after-registration' or not os.path.exists(REG) or 'osf.io' not in open(REG).read():
        sys.exit('Refusing to read test-phase spikes: registration URL missing or UNBLIND not set.')


def stop_if_nan(name, *vals):
    if not all(np.isfinite(v) for v in vals): sys.exit(f'NaN in {name}: stopping (rule written in the registration).')


def safe(fn, *a, **k):
    """Exploratory and robustness items: an error is recorded, never allowed to stop the run."""
    try: return fn(*a, **k)
    except Exception as ex: return dict(error=repr(ex))


def _copy(sessions, fn):
    """Sessions with a private copy of the test table, changed by fn."""
    out = []
    for s in sessions:
        c = dict(s); c['test'] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s['test'].items()}; fn(c); out.append(c)
    return out


# ---------------- 001187 ----------------
def test_counts_001187(path):
    """Spike counts of every neuron in TEST_WINDOW for every test trial."""
    f = h5py.File(path, 'r'); L = f['intervals/LTM_trials']; on = L['timestamps_PicOnset'][:]
    st = f['units/spike_times'][:]; idx = f['units/spike_times_index'][:].astype(np.int64); starts = np.r_[0, idx[:-1]]
    tc = np.zeros((len(idx), len(on)))
    for u in range(len(idx)):
        sp = np.sort(st[starts[u]:idx[u]])
        tc[u] = np.searchsorted(sp, on + A.TEST_WINDOW[1]) - np.searchsorted(sp, on + A.TEST_WINDOW[0])
    return tc


def stability_001187(path):
    """Neurons kept: waveform, rate and refractory violations stable from the encoding phase to the test phase, and
    >= 40 spikes in each. Rates = spikes in the phase / its duration. No image or answer information is used."""
    f = h5py.File(path, 'r'); U = f['units']; W = f['intervals/WM_trials']; L = f['intervals/LTM_trials']
    t1a, t1b = W['start_time'][0], W['stop_time'][-1]; t2a, t2b = L['start_time'][0], L['stop_time'][-1]
    st = U['spike_times'][:]; idx = U['spike_times_index'][:].astype(np.int64); starts = np.r_[0, idx[:-1]]; wf = U['waveforms']
    assert len(wf) == len(st) and np.array_equal(U['waveforms_index_index'][:].astype(np.int64), idx), 'waveform layout'
    keep = []
    for u in range(len(idx)):
        sp = st[starts[u]:idx[u]]; a = np.where((sp >= t1a) & (sp <= t1b))[0]; b = np.where((sp >= t2a) & (sp <= t2b))[0]
        if len(a) < 40 or len(b) < 40: keep.append(False); continue
        pick = lambda k: wf[starts[u] + k[np.linspace(0, len(k) - 1, min(len(k), 400)).astype(int)]].mean(0)
        w1, w2 = pick(a), pick(b); r1 = len(a) / (t1b - t1a); r2 = len(b) / (t2b - t2a)
        isi = lambda x: float((np.diff(np.sort(x)) < 0.003).mean()) if len(x) > 1 else 0.0
        ok = (np.corrcoef(w1, w2)[0, 1] >= WF_CORR and AMP_LO <= np.ptp(w2) / np.ptp(w1) <= AMP_HI and RATE_LO <= r2 / r1 <= RATE_HI
              and isi(sp[a]) <= ISI_MAX and isi(sp[b]) <= ISI_MAX)
        keep.append(bool(ok))
    return np.array(keep)


def prepare_001187():
    if not DRY: guard()
    A.category = load_001187.cat_of
    sessions = []; excl = {}; rng = np.random.default_rng(1); pf = {}
    for p in load_001187.files():
        s = load_001187.load_session(p)
        if DRY:
            s = simulate_session(s, {'corr': 1.0}, rng, pf); keep = np.ones(s['enc'].shape[0], bool)
        else:
            keep = stability_001187(p); s['test_count'] = test_counts_001187(p)
        excl[s['file']] = int((~keep).sum())
        s['enc'] = np.where(keep[:, None], s['enc'], 0.0); A.set_pop(s)
        s['tuned'] = [u for u in A.tuned_neurons(s) if keep[u]]; sessions.append(s)
    return sessions, excl


def decide(z):
    return 'positive' if z > 2.5 else ('negative' if z < -2.5 else 'not found')


def pairs_test(sessions, perm_seed, stop=False, **kw):
    """Pair swap test. stop=True (000004 test): a NaN stops the run; otherwise (exploratory) a NaN is recorded."""
    out = A.pairs(sessions, **kw); pv = A.pair_swap_test(out, np.random.default_rng(perm_seed), n=NDRAW)
    if stop: stop_if_nan('pairs ' + str(kw), pv['z'])
    dec = decide(pv['z']) if np.isfinite(pv['z']) else 'NaN (not used)'
    return dict(pv, decision=dec, n_B=sum(len(v['res']) for v in out.values()))


def repetition_test(sessions, stop=True):
    """Repetition test: z = k / SD of 100 crossed bootstrap draws, on the log scale, the identity scale and the log scale
    with memory terms. Found if |z_log| > 1.6 and all three z have the same sign. Primary (stop=True): a NaN stops the
    run; robustness (stop=False): a NaN is recorded."""
    rng = np.random.default_rng(20261120); out = {}
    for name, sc, mem in (('log', 'log', False), ('identity', 'identity', False), ('log_memory', 'log', True)):
        e, sd = A.boot_repetition(sessions, rng, n=100, scale=sc, memory=mem)
        out[name] = dict(k=float(e['k']), z=float(e['k'] / sd['k']), k_first_viewing=float(e['k_first_viewing']))
    if not all(np.isfinite(v['z']) for v in out.values()):
        if stop:
            os.makedirs(RES, exist_ok=True)
            open(os.path.join(RES, '001187_repetition_DRYRUN.json' if DRY else '001187_repetition.json'), 'w').write(json.dumps(dict(repetition=out, decision='not tested (z could not be computed)'), indent=1, default=float))
            stop_if_nan('repetition', *[v['z'] for v in out.values()])
        return dict(result=out, decision='NaN (not used)')
    z = out['log']['z']; same = np.sign(out['identity']['z']) == np.sign(z) == np.sign(out['log_memory']['z'])
    dec = ('negative' if z < -1.6 else 'positive' if z > 1.6 else 'not found') if same else 'not found'
    return dict(result=out, decision=dec)


def robustness_001187(sessions):
    """Descriptive only. Pairs: answers after 1.0 s; leaving out one participant (range); test window 0.2-1.0 s.
    Repetition: answers after 1.0 s; leaving out one participant (range); twice/once labels shuffled; second viewing in
    the same trial vs an earlier trial; 0.2-0.5 s window in both phases with all reaction times. A dry run runs the
    window checks on the synthetic counts."""
    r = {}
    def rt1(c): c['test']['keep_pairs'] = c['test']['keep_pairs'] & (c['test']['rt'] >= 1.0); c['test']['keep_repetition'] = c['test']['keep_repetition'] & (c['test']['rt'] >= 1.0)
    r['pairs_rt_ge_1s'] = safe(pairs_test, _copy(sessions, rt1), SEED_1187 + 10); r['repetition_rt_ge_1s'] = safe(repetition_test, _copy(sessions, rt1), stop=False)
    def loo_pairs():
        out = A.pairs(sessions); parts = sorted(set(v['participant'] for v in out.values()))
        loo = [A.pair_stats({k: v for k, v in out.items() if v['participant'] != pj})['best'] for pj in parts]
        return dict(min=float(np.nanmin(loo)), max=float(np.nanmax(loo)))
    def loo_repetition():
        parts = sorted(set(s['participant'] for s in sessions))
        v = [float(A.fit_repetition(A.all_repetition_rows(sessions, pj))['k']) for pj in parts]
        return dict(min=float(np.nanmin(v)), max=float(np.nanmax(v)))
    r['pairs_leave_one_out'] = safe(loo_pairs); r['repetition_leave_one_out'] = safe(loo_repetition)
    r['repetition_label_shuffle'] = safe(A.label_shuffle, sessions)
    for kind, name in ((1, 'repetition_probe_same_trial'), (2, 'repetition_probe_earlier_trial')):
        def only(c, kind=kind): c['test']['keep_repetition'] = c['test']['keep_repetition'] & ~((c['test']['shown'] == 2) & (c['test']['probe_kind'] != kind))
        r[name] = safe(repetition_test, _copy(sessions, only), stop=False)
    for win, name in (((0.2, 1.0), 'pairs_window_0.2_1.0'), ((0.2, 0.5), 'repetition_window_0.2_0.5')):
        def one(win=win, name=name):
            old_w = A.TEST_WINDOW; A.TEST_WINDOW = win
            try:
                ss = []
                for s, p in zip(sessions, load_001187.files()):
                    c = dict(s); c['test'] = dict(s['test']); c['test_count'] = s['test_count'] if DRY else test_counts_001187(p)
                    if name.startswith('repetition'):             # the same window at the first viewing; all reaction times
                        c['enc_s'] = np.array([np.searchsorted(tr, s['enc_on'] + win[1]) - np.searchsorted(tr, s['enc_on'] + win[0]) for tr in s['trains']], float)
                        T = c['test']; T['keep_repetition'] = (T['conf'] >= 1) & (T['conf'] <= 6) & np.isin(T['shown'], [0, 1, 2]) & np.array([A.category(i) > 0 for i in T['id']])
                    ss.append(c)
                return pairs_test(ss, SEED_1187 + 11) if name.startswith('pairs') else repetition_test(ss, stop=False)
            finally:
                A.TEST_WINDOW = old_w
        r[name] = safe(one)
    return r


def counts_001187(sessions):
    """Participants, sessions, neurons; trials dropped for fast answers per image group; delay quartiles once vs twice."""
    c = dict(participants=len(set(s['participant'] for s in sessions)), sessions=len(sessions), tuned_neurons=int(sum(len(s['tuned']) for s in sessions)),
             active_neurons=int(sum((s['enc'].sum(1) >= 20).sum() for s in sessions)))
    for g, name in ((0, 'new'), (1, 'once'), (2, 'twice')):
        c[f'fast_answers_dropped_{name}'] = int(sum(((s['test']['shown'] == g) & (s['test']['rt'] < A.MIN_RT)).sum() for s in sessions))
    d1 = np.concatenate([s['test']['delay'][s['test']['keep_repetition'] & (s['test']['shown'] == 1)] for s in sessions])
    d2 = np.concatenate([s['test']['delay'][s['test']['keep_repetition'] & (s['test']['shown'] == 2)] for s in sessions])
    c['log_delay_once_quartiles'] = np.nanpercentile(d1, [25, 50, 75]).tolist(); c['log_delay_twice_quartiles'] = np.nanpercentile(d2, [25, 50, 75]).tolist()
    return c


def run_001187():
    sessions, excl = prepare_001187(); os.makedirs(RES, exist_ok=True)
    res = dict(dataset='001187', dry_run=DRY, excluded_neurons=excl, repetition=repetition_test(sessions))
    first = json.dumps(dict(res, note='preregistered test, written before any exploratory analysis'), indent=1, default=float)
    open(os.path.join(RES, '001187_repetition_DRYRUN.json' if DRY else '001187_repetition.json'), 'w').write(first)   # kept even if a later step fails
    print('repetition SHA-256', hashlib.sha256(first.encode()).hexdigest())
    res['pairs'] = safe(pairs_test, sessions, SEED_1187, phase='enc')
    res['pairs_controls'] = dict(pre=safe(pairs_test, sessions, SEED_1187 + 1, phase='pre'), shuffle=safe(pairs_test, sessions, SEED_1187 + 2, shuffle=True, seed=7))
    res['pairs_interpretable'] = all('z' in c and abs(c['z']) <= 2.5 for c in res['pairs_controls'].values())   # NaN or error: not interpretable
    res['pairs_secondary'] = dict(hold=safe(pairs_test, sessions, SEED_1187 + 3, phase='hold'),
                                  hippocampus=safe(pairs_test, sessions, SEED_1187 + 5, b_area='hippocampus'),
                                  amygdala=safe(pairs_test, sessions, SEED_1187 + 6, b_area='amygdala'))
    res['robustness'] = robustness_001187(sessions)
    res['counts'] = safe(counts_001187, sessions)
    js = json.dumps(res, indent=1, default=float)
    open(os.path.join(RES, '001187_DRYRUN.json' if DRY else '001187.json'), 'w').write(js)
    print(js[:3000]); print('SHA-256', hashlib.sha256(js.encode()).hexdigest())


# ---------------- 000004 ----------------
def stability_000004(path):
    """As stability_001187, from the mean waveforms stored for each phase; there is no 40-spike rule."""
    f = h5py.File(path, 'r'); U = f['units']; t = f['intervals/trials']
    ph = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in t['stim_phase'][:]]); on = t['stim_on_time'][:]
    la, lb = on[ph == 'learn'].min(), on[ph == 'learn'].max() + 3; ra, rb = on[ph == 'recog'].min(), on[ph == 'recog'].max() + 3
    w1 = U['waveform_mean_encoding'][:]; w2 = U['waveform_mean_recognition'][:]
    st = U['spike_times'][:]; idx = U['spike_times_index'][:].astype(np.int64); starts = np.r_[0, idx[:-1]]; keep = []
    for u in range(len(idx)):
        sp = st[starts[u]:idx[u]]; r1 = ((sp >= la) & (sp <= lb)).sum() / (lb - la); r2 = ((sp >= ra) & (sp <= rb)).sum() / (rb - ra)
        isi = lambda x: float((np.diff(np.sort(x)) < 0.003).mean()) if len(x) > 1 else 0.0
        ok = (np.corrcoef(w1[u], w2[u])[0, 1] >= WF_CORR and AMP_LO <= np.ptp(w2[u]) / (np.ptp(w1[u]) + 1e-12) <= AMP_HI and r1 > 0
              and RATE_LO <= r2 / r1 <= RATE_HI and isi(sp[(sp >= la) & (sp <= lb)]) <= ISI_MAX and isi(sp[(sp >= ra) & (sp <= rb)]) <= ISI_MAX)
        keep.append(bool(ok))
    return np.array(keep)


def run_000004():
    import load_000004 as D
    os.makedirs(RES, exist_ok=True)
    if not DRY:
        guard()
        if not os.path.exists(POSTED): sys.exit('Post the 001187 result (SHA-256) to the registration first.')
        js = open(os.path.join(RES, '001187.json')).read()
        if hashlib.sha256(js.encode()).hexdigest() not in open(POSTED).read(): sys.exit('Posted SHA-256 does not match results/001187.json.')
    A.category = lambda i: D.CAT.get(i, 0)        # 000004 categories come from the file, not from the image ID
    sessions = []; rng = np.random.default_rng(2); pf = {}
    for p in D.files():
        s = D.load_session(p)
        if DRY: s = simulate_session(s, {'corr': 1.0}, rng, pf); keep = np.ones(s['enc'].shape[0], bool)
        else:
            keep = stability_000004(p); f = h5py.File(p, 'r'); t = f['intervals/trials']
            ph = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in t['stim_phase'][:]]); on = t['stim_on_time'][:][ph == 'recog']
            st = f['units/spike_times'][:]; idx = f['units/spike_times_index'][:].astype(np.int64); starts = np.r_[0, idx[:-1]]
            s['test_count'] = np.array([np.searchsorted(np.sort(st[a:b]), on + A.TEST_WINDOW[1]) - np.searchsorted(np.sort(st[a:b]), on + A.TEST_WINDOW[0]) for a, b in zip(starts, idx)], float)
        s['enc'] = np.where(keep[:, None], s['enc'], 0.0); A.set_pop(s); s['tuned'] = [u for u in A.tuned_neurons(s) if keep[u]]; sessions.append(s)
    r = pairs_test(sessions, SEED_0004, stop=True, phase='enc')
    j = json.load(open(os.path.join(RES, '001187.json'))) if not DRY else dict(pairs=dict(z=3.0, decision='positive'), pairs_interpretable=True)
    prim = j['pairs']; passed = prim.get('decision') in ('positive', 'negative') and j['pairs_interpretable']   # 001187 pair analysis and its controls passed
    r['confirmed'] = bool(np.sign(r['z']) == np.sign(prim['z']) and abs(r['z']) > 1.645) if passed else 'descriptive only (001187 pair analysis did not pass)'
    r['controls'] = dict(pre=pairs_test(sessions, SEED_0004 + 1, phase='pre'), shuffle=pairs_test(sessions, SEED_0004 + 2, shuffle=True, seed=8))
    js = json.dumps(r, indent=1, default=float)
    open(os.path.join(RES, '000004_DRYRUN.json' if DRY else '000004.json'), 'w').write(js); print(js[:2000])


if __name__ == '__main__':
    (run_000004 if sys.argv[1:] == ['000004'] else run_001187)()
