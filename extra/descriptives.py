"""Exploratory and descriptive, not preregistered. Writes results/extra_descriptives.json and
results/extra_probe_control.json.

Descriptives: neurons per area, tuned neurons per area and per participant, test trials and exclusions per image
group, rows entering the repetition model, missing delays, and recognition behaviour (hit rates, false alarms, d').

Probe control: the same x and the same kind of model, applied to the second viewing itself. Within the encoding phase,
each probe repeats an image seen seconds or minutes earlier. Rows are the first viewings and the probes of images that also appear at test (never-tested images define x and the
tuning selection, so neither their first viewings nor their probes are used), 0.2-0.7 s;
one fixed effect per neuron x category; terms: second viewing, second viewing x x, match probe, and the first-viewing context
(time in the phase, one-image trial, first image of a trial), each also x x. k_probe is second viewing x x. If x tracks
a neuron's tuning well enough to show a tuning-proportional change, k_probe should differ from zero here."""
import json, os, sys, collections
import numpy as np, h5py
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analysis as A, run

sessions, excl = run.prepare_001187()

# ---------------- descriptives ----------------
d = {}
d['neurons_total'] = int(sum(s['enc'].shape[0] for s in sessions))
d['neurons_excluded'] = int(sum(excl.values()))
area = collections.Counter(); tuned_area = collections.Counter(); per_part = collections.Counter()
for s in sessions:
    for a in s['area']: area[a.split('_')[0]] += 1
    for u in s['tuned']: tuned_area[s['area'][u].split('_')[0]] += 1; per_part[s['participant']] += 1
d['neurons_by_area'] = dict(area); d['tuned_by_area'] = dict(tuned_area)
d['tuned_sessions'] = int(sum(1 for s in sessions if s['tuned'])); d['tuned_participants'] = len(per_part)
d['tuned_per_participant'] = dict(median=float(np.median(list(per_part.values()))), min=min(per_part.values()), max=max(per_part.values()))
g = {0: 'new', 1: 'once', 2: 'twice'}; tri = {}
for k, name in g.items():
    tot = sum(int(((s['test']['shown'] == k) & np.array([A.category(i) > 0 for i in s['test']['id']])).sum()) for s in sessions)
    fast = sum(int(((s['test']['shown'] == k) & (s['test']['rt'] < A.MIN_RT)).sum()) for s in sessions)
    tri[name] = dict(trials=tot, answered_before_0_7s=fast)
d['test_trials'] = tri
R = A.all_repetition_rows(sessions)
old = R[:, 2] >= 1; test = R[:, 11] == 1
d['model_rows'] = dict(test_once=int((old & test & (R[:, 2] == 1)).sum()), test_twice=int((old & test & (R[:, 2] == 2)).sum()),
                       first_viewing=int((old & ~test).sum()), test_new=int((~old & test).sum()))
d['test_rows_missing_delay'] = int(np.isnan(R[old & test, 12]).sum())
nt = [min(int((s['never_tested'] & (s['enc_cat'] == c)).sum()) for c in range(1, 6)) for s in sessions if s['tuned']]
d['never_tested_per_category_min_over_categories'] = dict(median=float(np.median(nt)), min=min(nt), max=max(nt))
xa = np.array([v for s in sessions for u in s['tuned'] for v in A.usual_response(s, u).values()])
d['x_quartiles_analysed'] = [float(q) for q in np.percentile(xa, [25, 50, 75])]
# behaviour: per session, then pooled over sessions; d' with a 0.5 correction
beh = {k: [] for k in ('hit_once', 'hit_twice', 'false_alarm', 'dprime_once', 'dprime_twice')}
for s in sessions:
    T = s['test']; ok = (T['conf'] >= 1) & (T['conf'] <= 6)
    rate = lambda m: (T['said_old'][m & ok].sum() + 0.5) / ((m & ok).sum() + 1)
    h1, h2, fa = rate(T['shown'] == 1), rate(T['shown'] == 2), rate(T['shown'] == 0)
    beh['hit_once'].append(h1); beh['hit_twice'].append(h2); beh['false_alarm'].append(fa)
    beh['dprime_once'].append(norm.ppf(h1) - norm.ppf(fa)); beh['dprime_twice'].append(norm.ppf(h2) - norm.ppf(fa))
d['behaviour_session_mean'] = {k: float(np.mean(v)) for k, v in beh.items()}
dd = np.array(beh['dprime_twice']) - np.array(beh['dprime_once'])
d['behaviour_dprime_twice_minus_once'] = dict(mean=float(dd.mean()), sd=float(dd.std(ddof=1)), n_sessions=len(dd),
                                              sessions_higher=int((dd > 0).sum()))
json.dump(d, open(os.path.join(run.RES, 'extra_descriptives.json'), 'w'), indent=1)
print(json.dumps(d, indent=1))


# ---------------- probe control ----------------
def probe_rows(s, code):
    f = h5py.File(os.path.join(A.DATA, '001187', s['participant'], s['file']), 'r'); W = f['intervals/WM_trials']
    p_on = W['timestamps_Probe'][:]; p_id = W['PicIDs_Probe'][:].astype(int); load = W['loads'][:]
    p_rt = W['timestamps_Response'][:] - p_on; match = (W['probe_in_out'][:] == 1).astype(float)   # 1: probe was in the trial
    lo, hi = s['enc_on'].min(), s['enc_on'].max()
    p_t = np.clip((p_on - lo) / max(hi - lo, 1e-9), 0, 1) - 0.5
    nt_ids = set(s['enc_id'][s['never_tested']].tolist())
    rows = []
    for u in s['tuned']:
        xs = A.usual_response(s, u); sp = s['trains'][u]
        for j, i in enumerate(s['enc_id']):
            c = A.category(i)
            if c == 0 or s['never_tested'][j]: continue   # never-tested images define x: kept out of the baseline
            rows.append((u * 10 + c, s['enc_s'][u, j], 0, xs[c], s['enc_t'][j], float(s['enc_load'][j] == 1),
                         float(s['enc_k'][j] == 1), s['image_set'] * 10000 + i, code, 0.0))
        cnt = np.searchsorted(sp, p_on + A.TEST_WINDOW[1]) - np.searchsorted(sp, p_on + A.TEST_WINDOW[0])
        for j, i in enumerate(p_id):
            c = A.category(i)
            if c == 0 or i in nt_ids or not p_rt[j] >= A.MIN_RT: continue   # x images out; the window never contains the answer
            rows.append((u * 10 + c, cnt[j], 1, xs[c], p_t[j], float(load[j] == 1), 0.0, s['image_set'] * 10000 + i, code, match[j]))
    return np.array(rows, float)


parts_ = sorted(set(s['participant'] for s in sessions)); code = {k: j for j, k in enumerate(parts_)}; P = []
for j, s in enumerate(sessions):
    if s['tuned']:
        r = probe_rows(s, code[s['participant']]); r[:, 0] += 100000 * j; P.append(r)
P = np.concatenate(P)


def fit(P, w):
    sec, x = P[:, 2], P[:, 3]; ctx = P[:, 4:7] - P[:, 4:7].mean(0)
    X = np.column_stack([sec, sec * x, sec * P[:, 9], ctx, ctx * x[:, None]])
    b = A.pois_fe(P[:, 0], P[:, 1], X, np.zeros(len(P)), w)
    return b[0], b[1]


rng = np.random.default_rng(20261122)
part_of = P[:, 8].astype(int); pics, pic_of = np.unique(P[:, 7], return_inverse=True)
est = fit(P, np.ones(len(P))); bs = []
for _ in range(100):
    sc = np.bincount(rng.integers(0, len(parts_), len(parts_)), minlength=len(parts_))
    pc = np.bincount(rng.integers(0, len(pics), len(pics)), minlength=len(pics))
    bs.append(fit(P, (sc[part_of] * pc[pic_of]).astype(float)))
bs = np.array(bs)
out = dict(second=float(est[0]), second_sd=float(np.nanstd(bs[:, 0])), k_probe=float(est[1]), k_probe_sd=float(np.nanstd(bs[:, 1])),
           n_probe_rows=int((P[:, 2] == 1).sum()), n_first_rows=int((P[:, 2] == 0).sum()), n_nan=int(np.isnan(bs[:, 1]).sum()))
out['k_probe_z'] = out['k_probe'] / out['k_probe_sd']; out['second_z'] = out['second'] / out['second_sd']
json.dump(out, open(os.path.join(run.RES, 'extra_probe_control.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
