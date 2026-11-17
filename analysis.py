"""Analyses of the study, shared by the locked runner (run.py), the calibration on synthetic worlds (calibrate.py)
and the encoding-phase numbers disclosed in the preregistration (examined.py).

Repetition (the preregistered test): for tuned neurons, does the change of a neuron's response from the first viewing
to the test depend on x, its usual response to the image's category, more for images shown twice than once? The tested
coefficient is k.
Pairs (exploratory): for once-seen images, does neuron B's test response depend on the product of A's and B's
first-viewing responses, more than for pseudo-pairs (neurons of other participants that saw the same images)?

Sessions come from load_001187 or load_000004. Nothing here reads spikes."""
import os
import numpy as np

DATA = os.environ.get('DATA', '')          # folder holding 001187/ and 000004/
INPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs')
X_WINDOW = (0.2, 1.2)      # encoding-phase window in which x is measured (image on screen for 2 s)
TEST_WINDOW = (0.2, 0.7)   # test window, fixed; trials answered before its end are dropped
MIN_RT = 0.7
MIN_TRIALS = 30            # a session needs at least this many once-seen test trials for the pair analysis
PRE = (-0.5, 0.0)          # before the image: the participant's state without image drive (negative control)
HOLD = (0.2, 2.4)          # hold period after the last image of a trial (the probe comes ~2.6 s later); counts are
                           # scaled to the length of X_WINDOW so both periods are on the same scale
category = None            # image ID -> category (1-5; 0 = none) of the dataset in use; set by the calling script


def set_pop(sess):
    """Active neurons (>= 20 encoding-phase spikes) and each neuron's response to every presentation divided by its mean
    on never-tested images; used to measure the trial state."""
    enc = sess['enc']; good = enc.sum(1) >= 20
    nm = np.zeros(enc.shape); nm[good] = enc[good] / (enc[good][:, sess['never_tested']].mean(1, keepdims=True) + 1e-6)
    sess['_nm'] = nm; sess['_good'] = good


def tuned_neurons(sess):
    """Neurons whose encoding-phase response differs across categories (one-way ANOVA p < 0.05, never-tested images)."""
    from scipy.stats import f_oneway
    out = []; S = sess['never_tested']
    for u in range(sess['enc'].shape[0]):
        g = [sess['enc'][u, S & (sess['enc_cat'] == c)] for c in range(1, 6)]
        if sess['enc'][u].sum() < 20 or min(len(x) for x in g) < 3: continue
        if f_oneway(*g).pvalue < 0.05: out.append(u)
    return out


def usual_response(sess, u):
    """x for each category: the neuron's mean response to that category's never-tested images, divided by its mean
    response to all never-tested images, minus 1."""
    S = sess['never_tested']; e = sess['enc'][u]; m = e[S].mean() + 1e-6
    return {c: e[S & (sess['enc_cat'] == c)].mean() / m - 1 for c in range(1, 6)}


# ---------------- regression tools ----------------
def _demean(g, y, X, w):
    order = np.argsort(g, kind='stable')
    gs = g[order]; ys = y[order].copy(); Xs = X[order].copy(); ws = w[order]
    bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        ww = ws[a:b]; sw = ww.sum(); ys[a:b] -= (ww * ys[a:b]).sum() / sw; Xs[a:b] -= (ww[:, None] * Xs[a:b]).sum(0) / sw
    return ys, Xs, ws


def _wls(y, X, w):
    keep = X.std(0) > 0; sw = np.sqrt(w); b = np.zeros(X.shape[1])
    b[keep] = np.linalg.lstsq(X[:, keep] * sw[:, None], y * sw, rcond=None)[0]; return b


def pois_fe(g, y, X, offset, w, iters=30):
    """Poisson regression (log link) with one fixed effect per group g, profiled out. Returns coefficients of X;
    NaN if the fit does not converge."""
    _, gi = np.unique(g, return_inverse=True)
    sy = np.bincount(gi, w * y)
    ok = sy[gi] > 0; gi = np.unique(gi[ok], return_inverse=True)[1]
    y, X, offset, w = y[ok], X[ok], offset[ok], w[ok]
    keep = X.std(0) > 0; Xk = X[:, keep]
    b = np.zeros(Xk.shape[1]); sy = np.bincount(gi, w * y)
    for _ in range(iters):
        eta = Xk @ b + offset
        alpha = np.log(sy) - np.log(np.bincount(gi, w * np.exp(eta)))
        mu = np.exp(eta + alpha[gi]); wm = w * mu
        sw = np.bincount(gi, wm)
        Xbar = np.stack([np.bincount(gi, wm * Xk[:, k]) for k in range(Xk.shape[1])], 1) / sw[:, None]
        Xt = Xk - Xbar[gi]
        step = np.linalg.solve(Xt.T @ (Xt * wm[:, None]) + 1e-9 * np.eye(Xk.shape[1]), Xt.T @ (w * (y - mu)))
        b += step
        if np.abs(step).max() < 1e-7: break
    else:
        if not np.all(np.isfinite(b)) or np.abs(step).max() > 1e-4: b = b * np.nan
    full = np.zeros(len(keep)); full[keep] = b
    return full


def _cov(R):
    """Test covariates: position in the test (cubic), previous image old / called old, typicality and typicality * x."""
    o = R[:, 5]
    return np.column_stack([o, o ** 2 - 1 / 12, o ** 3, R[:, 6], R[:, 7], R[:, 8], R[:, 8] * R[:, 3]])


def _conf_dummies(conf):
    return np.column_stack([(conf == k).astype(float) for k in (2, 3, 4, 5, 6)])


def _conf_dev(conf):
    """Confidence within an answer: 1, 2 relative to 3 (said new); 5, 6 relative to 4 (said old)."""
    return np.column_stack([(conf == k).astype(float) for k in (1, 2, 5, 6)])


# ---------------- answer signal ----------------
def _neuron_d(gi, ni, n_neurons, y, so, eta, w, iters=60):
    """Each neuron's said-old signal d (log ratio), with one effect per group (neuron x category) and fixed other terms eta."""
    d = np.zeros(n_neurons); sy = np.bincount(gi, w * y)
    for _ in range(iters):
        e = eta + d[ni] * so
        alpha = np.log(sy + 1e-12) - np.log(np.bincount(gi, w * np.exp(e)))
        mu = np.exp(alpha[gi] + e)
        num = np.bincount(ni, w * y * so, minlength=n_neurons); den = np.bincount(ni, w * mu * so, minlength=n_neurons)
        step = np.where(den > 0, np.log((num + 1e-9) / (den + 1e-9)), 0.0); d += step
        if np.abs(step).max() < 1e-6: break
    return d, num


def answer_signal(g, neuron, y, so, P, w):
    """On new images: log rate = group + d_neuron * said_old + P b. Each neuron's own d is shrunk toward the pooled
    signal (empirical Bayes). The noise in each neuron's d is measured, not assumed: d is measured in the first and
    second half of the neuron's trials, and their disagreement gives the noise. Real differences between neurons
    (decision neurons) stay."""
    nu, ni = np.unique(neuron, return_inverse=True); _, gi = np.unique(g, return_inverse=True); nn = len(nu)
    pooled_fit = pois_fe(g, y, np.column_stack([so, P]), np.zeros(len(y)), w)
    pooled, b = pooled_fit[0], pooled_fit[1:]
    eta = P @ b
    d, n1 = _neuron_d(gi, ni, nn, y, so, eta, w)
    order = np.argsort(ni, kind='stable'); rank = np.empty(len(ni), int); rank[order] = np.arange(len(ni))
    first_of = np.r_[0, np.cumsum(np.bincount(ni, minlength=nn))[:-1]]; n_of = np.bincount(ni, minlength=nn)
    half = ((rank - first_of[ni]) >= n_of[ni] / 2).astype(int)   # first vs second half of the session (rows are in time
                                                                  # order), so slow drift counts as noise
    dh = []
    for h in (0, 1):
        m = half == h; _, gih = np.unique(g[m], return_inverse=True)
        dh.append(_neuron_d(gih, ni[m], nn, y[m], so[m], eta[m], w[m])[0])
    ok = (n1 > 0) & np.isfinite(dh[0]) & np.isfinite(dh[1]) & np.isfinite(d)
    noise_var = np.mean((dh[0][ok] - dh[1][ok]) ** 2) / 4 if ok.any() else np.inf       # variance of the full-data d
    tau2 = max(0.0, float(np.var(d[ok]) - noise_var)) if ok.any() else 0.0
    lam = np.where(ok, tau2 / (tau2 + noise_var), 0.0)
    d = pooled + lam * (np.where(ok, d, pooled) - pooled)
    return nu, d, b


def answer_adjustment(R, w):
    """Answer signal measured on new images (each neuron's said-old signal + confidence within answer + covariates);
    returns the log-scale amount to remove from each row, so old images are never selected by answer."""
    g, y, once, x, conf = R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4]; so = (conf >= 4).astype(float); neuron = g // 10
    C = _cov(R); E = _conf_dev(conf); P = np.column_stack([E, C])
    mm = (once == 0) & (w > 0)
    nu, d, bp = answer_signal(g[mm], neuron[mm], y[mm], so[mm], P[mm], w[mm])
    dd = np.zeros(len(R)); pos = np.searchsorted(nu, neuron); ok = (pos < len(nu)) & (nu[np.minimum(pos, len(nu) - 1)] == neuron)
    dd[ok] = d[pos[ok]]
    return dd * so + E @ bp[:4]


# ---------------- repetition ----------------
def repetition_rows(sess, neurons, participant_code):
    """Rows for the repetition test. For every tuned neuron and every once/twice-seen test image: one row with the test
    count and one row with the first-viewing count of the same image, both in TEST_WINDOW. New-image test rows are kept
    for the answer signal. Columns: 0 group (neuron x category x phase), 1 count, 2 showings (0/1/2), 3 x, 4 confidence,
    5 test position, 6 previous image old, 7 previous image called old, 8 typicality, 9 image, 10 participant,
    11 phase (1 = test), 12 log delay, 13 log reaction time, 14-16 first-viewing context (time in the encoding phase,
    one-image trial, first image of its trial; 0 on test rows). Old test images without a first viewing are left out."""
    T = sess['test']; rows = []; fi = {}
    for k, i in enumerate(sess['enc_id']): fi.setdefault(i, k)
    for u in neurons:
        xs = usual_response(sess, u)
        for t in np.where(T['keep_repetition'])[0]:
            i = T['id'][t]; c = category(i); sh = int(T['shown'][t]); x = xs[c]
            if sh >= 1 and i not in fi: continue
            base = (sh, x, T['conf'][t], T['order'][t], T['prev_old'][t], T['prev_said'][t], T['typicality'][t],
                    sess['image_set'] * 10000 + i, participant_code)
            lrt = np.log(max(T['rt'][t], 0.1))
            rows.append((u * 100 + c * 10 + 1, sess['test_count'][u, t]) + base + (1, T['delay'][t], lrt, 0.0, 0.0, 0.0))
            if sh >= 1:
                j = fi[i]
                rows.append((u * 100 + c * 10 + 0, sess['enc_s'][u, j]) + base +
                            (0, 0.0, 0.0, sess['enc_t'][j], float(sess['enc_load'][j] == 1), float(sess['enc_k'][j] == 1)))
    return np.array(rows, float)


def all_repetition_rows(sessions, leave_out=None):
    """Repetition rows of every session, optionally without one participant's sessions; groups made unique per session."""
    keep = [s for s in sessions if s['participant'] != leave_out]
    parts_ = sorted(set(s['participant'] for s in keep)); code = {k: j for j, k in enumerate(parts_)}; parts = []
    for j, s in enumerate(keep):
        r = repetition_rows(s, s['tuned'], code[s['participant']])
        if len(r): r[:, 0] += 1000000 * j; parts.append(r)
    return np.concatenate(parts)


def fit_repetition(R, w=None, scale='log', memory=False):
    """The repetition test as a difference of differences on old images, both phases, with effects per neuron x
    category x phase. twice, twice*x (and twice x category) describe how twice-shown images differ at the first viewing
    (a property of the images); test x twice and test x twice*x describe how that difference changes by the test.
    k is test x twice*x. The answer signal (measured on new images) is removed from test rows. Test rows also get the
    test covariates, log time since the last viewing, log reaction time and test position, each also times x.
    First-viewing rows get their context (time in the encoding phase, one-image trial, first image of a trial) and its
    product with x: twice- and once-seen images were first seen in different contexts, and a context effect at the
    first viewing must not be carried into the test by the subtraction. memory=True adds said-old*x and
    high-confidence*x on test rows (twice-seen images are remembered better). scale 'log': Poisson; 'identity': least
    squares on the count divided by its group mean."""
    if w is None: w = np.ones(len(R))
    m0 = w > 0; R = R[m0]; w = w[m0]
    test = R[:, 11] == 1
    Rt = R[test].copy(); Rt[:, 2] = (Rt[:, 2] >= 1).astype(float); Rt[:, 0] = np.round(Rt[:, 0]) // 10   # group = neuron x category
    ans_t = answer_adjustment(Rt, w[test]); ans = np.zeros(len(R)); ans[test] = ans_t
    old = R[:, 2] >= 1; R = R[old]; w = w[old]; ans = ans[old]; test = R[:, 11] == 1
    g, y, x = R[:, 0], R[:, 1], R[:, 3]; tw = (R[:, 2] == 2).astype(float); ph = test.astype(float)
    cat = (np.round(g) // 10) % 10
    K = np.column_stack([tw * (cat == c) for c in range(2, 6)])
    cov = _cov(R) * ph[:, None]; dl = np.nan_to_num(R[:, 12], nan=np.nanmean(R[test, 12])); d = np.where(test, dl - dl[test].mean(), 0.0)
    lr = np.where(test, R[:, 13] - R[test, 13].mean(), 0.0)
    fr = ~test; ctx = np.zeros((len(R), 3)); ctx[fr] = R[fr, 14:17] - R[fr, 14:17].mean(0)
    cols = [ph * tw, ph * tw * x, tw, tw * x, K, K * ph[:, None], cov, d, d * x, lr, lr * x,
            ctx, ctx * x[:, None], ph * R[:, 5] * x]
    if memory:
        so = (R[:, 4] >= 4) * ph; hi = np.isin(R[:, 4], [5, 6]) * ph
        cols += [so * x, hi * x]
    X = np.column_stack(cols)
    if scale == 'log':
        b = pois_fe(g, y, X, -ans, w)
    else:
        yr = y * np.exp(-ans); gm = np.zeros(len(R))
        for gu in np.unique(g):
            mm = g == gu; gm[mm] = (w[mm] * yr[mm]).sum() / max(w[mm].sum(), 1e-9)
        yd, Xd, wd = _demean(g, yr / np.maximum(gm, 1e-6), X, w); b = _wls(yd, Xd, wd)
    return dict(twice=b[0], k=b[1], k_first_viewing=b[3])


def boot_repetition(sessions, rng, n=200, scale='log', memory=False):
    """Repetition estimate and its SD over a crossed bootstrap of participants and images (which images were shown twice
    is the same for every participant of an image set)."""
    R = all_repetition_rows(sessions); parts_ = sorted(set(s['participant'] for s in sessions))
    part_of = R[:, 10].astype(int); pics, pic_of = np.unique(R[:, 9], return_inverse=True)
    est = fit_repetition(R, scale=scale, memory=memory); bs = []
    for _ in range(n):
        sc = np.bincount(rng.integers(0, len(parts_), len(parts_)), minlength=len(parts_))
        pc = np.bincount(rng.integers(0, len(pics), len(pics)), minlength=len(pics))
        bs.append(fit_repetition(R, (sc[part_of] * pc[pic_of]).astype(float), scale=scale, memory=memory))
    sd = {key: float(np.std([b[key] for b in bs])) for key in est}
    return est, sd


def label_shuffle(sessions, n=200, seed=20261121):
    """k against k with the twice/once labels of the old images shuffled within image set and category (the same
    shuffle for every participant). Measures the null spread in the data itself, which the bootstrap z does not.
    Approximate: time since the last viewing keeps the true labels."""
    R = all_repetition_rows(sessions); rng = np.random.default_rng(seed); est = float(fit_repetition(R)['k'])
    img = R[:, 9]; cat = (np.round(R[:, 0]) // 10) % 10; old = R[:, 2] >= 1
    ids = {}
    for i, c, s in zip(img[old], cat[old], R[old, 2]): ids[i] = (int(i // 10000), int(c), s)
    groups = {}
    for i, (v, c, s) in ids.items(): groups.setdefault((v, c), []).append(i)
    null = []
    for _ in range(n):
        lab = {}
        for g in groups.values():
            sv = [ids[i][2] for i in g]; pv = rng.permutation(sv); lab.update(zip(g, pv))
        Rp = R.copy(); Rp[old, 2] = [lab[i] for i in img[old]]
        null.append(float(fit_repetition(Rp)['k']))
    null = np.array(null); ok = np.isfinite(null)
    return dict(k=est, null_sd=float(np.std(null[ok])), p_two=float((np.abs(null[ok]) >= abs(est)).mean()), n=n, n_nan=int((~ok).sum()))


# ---------------- pairs ----------------
def duplicates(sess, neurons, ms=0.001):
    """Pairs whose spikes coincide within +-1 ms far more than in the +-10..50 ms flanks (one neuron recorded twice)."""
    bad = set()
    for a in neurons:
        for b in range(len(sess['trains'])):
            if a == b: continue
            ta, tb = sess['trains'][a], sess['trains'][b]
            if len(ta) < 20 or len(tb) < 20: continue
            near = (np.searchsorted(tb, ta + ms) - np.searchsorted(tb, ta - ms)).sum()
            flank = (np.searchsorted(tb, ta + 0.05) - np.searchsorted(tb, ta + 0.01)).sum() + \
                    (np.searchsorted(tb, ta - 0.01) - np.searchsorted(tb, ta - 0.05)).sum()
            if near > 3 * flank * (2 * ms) / 0.08 + 5: bad.add((a, b)); bad.add((b, a))
    return bad


def state_residuals(sess):
    """Each neuron's first-viewing response with the trial state taken out. The state (attention, arousal) multiplies
    every neuron's response, so this works on the log scale: within each category, regress log(1 + count) on the log
    population response of that presentation (all active neurons except this one) and keep the residual."""
    enc = sess['enc']; nm = sess['_nm']; good = sess['_good']; cat = sess['enc_cat']
    tot = nm[good].sum(0); cnt = good.sum(); res = np.zeros(enc.shape)
    for u in range(enc.shape[0]):
        pop = (tot - (nm[u] if good[u] else 0)) / max(cnt - (1 if good[u] else 0), 1)
        lp = np.log(pop + 0.05); x = np.log1p(enc[u].astype(float)); xr = x.copy(); pr = lp.copy()
        for c in np.unique(cat):
            m = cat == c; xr[m] -= x[m].mean(); pr[m] -= lp[m].mean()
        b = (xr @ pr) / (pr @ pr) if (pr @ pr) > 0 else 0.0
        res[u] = x - b * pr
    sess['_res'] = res


def state_residuals_phase(sess, phase):
    """As state_residuals, for one encoding-phase period ('enc' image on screen, 'hold' after the last image of a trial,
    'pre' before the image), on the presentations that have it."""
    if phase == 'enc':
        state_residuals(sess); sess['_resP'] = sess['_res']; sess['_nmP'] = sess['_nm']; return
    M = sess[phase]; cols = np.isfinite(M).all(0); good = sess['_good']
    res = np.full(M.shape, np.nan); Mc = M[:, cols].astype(float)
    nm = np.zeros(Mc.shape); mu_ = Mc.mean(1, keepdims=True) + 1e-6; nm[good] = Mc[good] / mu_[good]
    tot = nm[good].sum(0); cnt = good.sum(); cat = sess['enc_cat'][cols]
    for u in range(M.shape[0]):
        pop = (tot - (nm[u] if good[u] else 0)) / max(cnt - (1 if good[u] else 0), 1)
        lp = np.log(pop + 0.05); x = np.log1p(Mc[u]); xr = x.copy(); pr = lp.copy()
        for c in np.unique(cat):
            m = cat == c; xr[m] -= x[m].mean(); pr[m] -= lp[m].mean()
        b = (xr @ pr) / (pr @ pr) if (pr @ pr) > 0 else 0.0
        res[u, cols] = x - b * pr
    full_nm = np.full(M.shape, np.nan); full_nm[:, cols] = nm
    sess['_resP'] = res; sess['_nmP'] = full_nm


def _first_index(sess):
    d = {}
    for j, i in enumerate(sess['enc_id']): d.setdefault(i, j)
    return d


def pair_betas(sessB, B, Z_A, POP, POPT=None):
    """Z_A: (n_A x n_trials) first-viewing responses of the A neurons to each test trial's image. Returns, for each A,
    the coefficient of A x B (the last column) in a regression of B's test response. POPT=None leaves out the test
    response of the other neurons (the form of the encoding-phase pilot in examined.py)."""
    T = sessB['test']; idx = sessB['_pair_trials']; fi = sessB['_first']
    S = sessB['never_tested']; eB = sessB['enc'][B]; mB = eB[S].mean() + 1e-6
    catm = np.array([eB[S & (sessB['enc_cat'] == c)].mean() for c in range(1, 6)])
    W = TEST_WINDOW[1] - TEST_WINDOW[0]
    y = sessB['test_count'][B, idx] / W / mB - catm[np.array([category(i) for i in T['id'][idx]]) - 1] / mB
    cats = np.array([category(i) for i in T['id'][idx]])
    zB = np.array([sessB['_res'][B, fi[i]] for i in T['id'][idx]], float)          # B's first response, trial state removed
    for c in range(1, 6):                                                          # and beyond its mean for that category
        mc = cats == c
        if mc.any(): zB[mc] -= zB[mc].mean()
    if zB.std() == 0: return np.full(len(Z_A), np.nan)
    b_ = (zB - zB.mean()) / zB.std()
    K = np.column_stack([(cats == c).astype(float) for c in range(2, 6)])          # category-level changes of B are absorbed
    # B's own curve (the later response is not a straight line in the first one), and B's own response re-expressed
    # on 'old' answers (replay)
    b2 = b_ ** 2 - (b_ ** 2).mean(); b3 = b_ ** 3 - (b_ ** 3).mean()
    so = (T['conf'][idx] >= 4).astype(float)
    C = np.column_stack([np.ones(len(idx)), K, _conf_dummies(T['conf'][idx]), b_, b2, b3, so * b_, so * b2])
    Z_A = np.array(Z_A, float)                                                     # copy: the pseudo set is reused
    for c in range(1, 6):                                                          # A's response beyond its mean for that category
        mc = cats == c
        if mc.any(): Z_A[:, mc] -= Z_A[:, mc].mean(1, keepdims=True)
    sd = Z_A.std(1, keepdims=True)
    a = (Z_A - Z_A.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1)
    POPc = POP - POP.mean(1, keepdims=True)                                       # trial state, and trial state x B
    # A itself; the test response of the other neurons to that image (without A and B), and its product with B, which
    # remove what the participant's own meaning of the image adds at test
    extra = [a[:, :, None]]
    if POPT is not None:
        PT = POPT - POPT.mean(1, keepdims=True); extra += [PT[:, :, None], (PT * b_[None, :])[:, :, None]]
    X = np.concatenate([np.broadcast_to(C, (len(a),) + C.shape), POP[:, :, None], (POPc * b_[None, :])[:, :, None]] + extra +
                       [(a * b_[None, :])[:, :, None]], 2)
    XtX = np.einsum('kni,knj->kij', X, X) + 1e-8 * np.eye(X.shape[2]); Xty = np.einsum('kni,n->ki', X, y)
    beta = np.linalg.solve(XtX, Xty[:, :, None])[:, -1, 0]
    beta[sd[:, 0] == 0] = np.nan
    return beta


def pairs(sessions, phase='enc', shuffle=False, seed=0, b_area=None, test_population=True):
    """Pair analysis. For each neuron B (>= 20 encoding-phase spikes) of a session with >= MIN_TRIALS once-seen test
    trials: real set = neurons A of B's session (>= 20 spikes) on another electrode and another location label than B,
    not duplicates; pseudo sets = for each session of the same image set from another participant, its neurons
    (>= 20 spikes) at another location label than B. Each A gives the A x B coefficient of pair_betas, ranked u among
    the pooled pseudo coefficients (a pseudo set without itself). A set's score is q = (max u) ** n, n = size of the
    real set; each pseudo set is cut to n random neurons so the strongest partner is compared between sets of equal
    size (sets smaller than n are skipped). v = rank of the real q among the pseudo sets' q (ties count 0.5). B is
    skipped if fewer than 20 pseudo coefficients or 5 pseudo sets remain; a session is skipped if fewer than 5 other
    sessions qualify. phase: 'enc' (image on screen), 'hold' (after the last image of a trial; only images followed
    by one), 'pre' (0.5 s before the image; first images of a trial only, so no earlier image is inside the window).
    shuffle (negative control): each tested image takes the encoding-phase responses of another image of its category.
    b_area: only B neurons whose location label starts with it. test_population=False: without the test response of
    the other neurons as a covariate (the form of the encoding-phase pilot)."""
    out = {}
    for s in sessions:
        s['_Acand'] = [u for u in range(s['enc'].shape[0]) if s['enc'][u].sum() >= 20]
        state_residuals_phase(s, phase); s['_res'] = s['_resP']
        T = s['test']; s['_first'] = _first_index(s)
        if shuffle:   # the same substitute for every session of an image set
            rs = np.random.default_rng(seed + 7919 * s['image_set'])
            ids = sorted(s['_first']); newf = {}
            for c in sorted(set(category(i) for i in ids)):
                grp = [i for i in ids if category(i) == c]
                if len(grp) < 2: continue
                perm = rs.permutation(len(grp))
                while (perm == np.arange(len(grp))).any(): perm = rs.permutation(len(grp))
                for j, i in enumerate(grp): newf[i] = s['_first'][grp[perm[j]]]
            s['_first'] = {i: newf.get(i, j) for i, j in s['_first'].items()}
        okt = [t for t in np.where(T['keep_pairs'] & (T['shown'] == 1))[0] if T['id'][t] in s['_first']]
        if phase == 'hold':
            okt = [t for t in okt if np.isfinite(s['_resP'][0, s['_first'][T['id'][t]]])]
        if phase == 'pre':
            okt = [t for t in okt if s['enc_k'][s['_first'][T['id'][t]]] == 1]
        s['_pair_trials'] = np.array(okt, int)
    for s in sessions:
        T = s['test']; idx = s['_pair_trials']; pics = T['id'][idx]; fi = s['_first']
        pres = np.array([fi[i] for i in pics], int)
        dup = duplicates(s, s['_Acand'])
        others = [o for o in sessions if o['image_set'] == s['image_set'] and o['participant'] != s['participant'] and len(o['_Acand']) >= 2
                  and all(i in o['_first'] for i in pics)]           # the other session must have shown every one of these images
        if len(others) < 5: out[s['file']] = dict(participant=s['participant'], res=[]); continue
        Zp = np.concatenate([o['_res'][o['_Acand']][:, [o['_first'][i] for i in pics]] for o in others]).astype(float)
        finite = np.isfinite(Zp).all(1)
        set_of = np.concatenate([np.full(len(o['_Acand']), j) for j, o in enumerate(others)])
        area_p = np.array([o['area'][a] for o in others for a in o['_Acand']])
        Zp, set_of, area_p = Zp[finite], set_of[finite], area_p[finite]
        if len(idx) < MIN_TRIALS: out[s['file']] = dict(participant=s['participant'], res=[]); continue
        Zr_all = {a: s['_res'][a, pres].astype(float) for a in s['_Acand']}
        tc = s['test_count']; s['_tn'] = tc / (tc[:, T['keep_pairs']].mean(1, keepdims=True) + 1e-6)   # test responses / neuron's test mean
        res = []
        for B in [u for u in range(s['enc'].shape[0]) if s['enc'][u].sum() >= 20
                  and (b_area is None or s['area'][u].startswith(b_area))]:
            As = [a for a in s['_Acand'] if a != B and s['elec'][a] != s['elec'][B] and (a, B) not in dup and s['area'][a] != s['area'][B]]
            if not As: continue
            # trial state: encoding-phase population response of that presentation, from active neurons except B and
            # except one neuron (the real A itself, or a random neuron of B's session for a pseudo A)
            m = s['_good'].copy(); m[B] = False
            if m.sum() < 3: continue
            NM = s['_nmP'][:, pres]; tot = NM[m].sum(0); cnt = m.sum()
            loo = lambda c: (tot - (NM[c] if m[c] else 0)) / (cnt - (1 if m[c] else 0))
            POPr = np.array([loo(a) for a in As])
            rnd = np.random.default_rng(B).choice(np.where(m)[0], len(Zp))
            POPp = (tot[None, :] - NM[rnd]) / (cnt - 1)
            # the same leave-out rule for the test phase
            TN = s['_tn'][:, idx]; ttot = TN[m].sum(0)
            POPTr = np.array([(ttot - (TN[a] if m[a] else 0)) / (cnt - (1 if m[a] else 0)) for a in As])
            POPTp = (ttot[None, :] - TN[rnd]) / (cnt - 1)
            if not test_population: POPTr = POPTp = None
            r = pair_betas(s, B, np.array([Zr_all[a] for a in As]), POPr, POPTr)
            p = pair_betas(s, B, Zp, POPp, POPTp)
            okp = ~np.isnan(p) & (area_p != s['area'][B])
            r = r[~np.isnan(r)]
            if not len(r) or okp.sum() < 20: continue
            p, so = p[okp], set_of[okp]
            ps = np.sort(p); N = len(p); n = len(r)
            u_r = np.searchsorted(ps, r) / N
            rk = np.random.default_rng(1000 + B)
            q_p = []; q_set = []
            for j in np.unique(so):
                x = p[so == j]
                if len(x) < n: continue
                x = x[rk.permutation(len(x))[:n]]; own = np.sort(p[so == j])
                u = (np.searchsorted(ps, x) - np.searchsorted(own, x)) / (N - (so == j).sum())
                q_p.append(u.max() ** n); q_set.append(others[j]['file'])
            if len(q_p) < 5: continue
            q_p = np.array(q_p); q_r = u_r.max() ** n
            v = ((q_p < q_r).sum() + 0.5 * (q_p == q_r).sum()) / len(q_p)
            res.append(dict(B=B, u=u_r, v=v, q_r=q_r, q_p=q_p, q_set=q_set))
        out[s['file']] = dict(participant=s['participant'], res=res)
    return out


def _participant_mean(vals_by_session, participant_of):
    """Mean over B within a session, then over sessions within a participant, then over participants."""
    per_p = {}
    for key, v in vals_by_session.items():
        if len(v): per_p.setdefault(participant_of[key], []).append(np.mean(v))
    return float(np.mean([np.mean(x) for x in per_p.values()])) if per_p else np.nan


def pair_stats(out, keys=None):
    """best (the pair statistic) = mean of v - 0.5; rank = the same for every pair's rank u."""
    keys = list(out) if keys is None else keys
    part = {key: out[key]['participant'] for key in keys}
    best = _participant_mean({key: [r['v'] for r in out[key]['res']] for key in keys}, part)
    rank = _participant_mean({key: [r['u'].mean() for r in out[key]['res']] for key in keys}, part)
    return dict(best=best - 0.5, rank=rank - 0.5)


def pair_swap_test(out, rng, n=1000):
    """Swap test for the pair analysis. Null draw: in every session, one other participant's set plays 'real' (the same
    one for every B of that session), ranked among the remaining sets and the real one, exactly as the real set is
    ranked. This keeps how strongly the neurons of one session resemble each other. z = statistic / SD of the null draws."""
    real = pair_stats(out)['best']
    sess = [(key, v['res']) for key, v in out.items() if v['res']]
    part = {key: out[key]['participant'] for key, _ in sess}
    sets = {key: sorted(set(x for r in res for x in r['q_set'])) for key, res in sess}
    null = []
    for _ in range(n):
        vals = {}
        for key, res in sess:
            pick = sets[key][rng.integers(len(sets[key]))]; vs = []
            for r in res:
                idx = [i for i, sj in enumerate(r['q_set']) if sj == pick]
                j = idx[0] if idx else rng.integers(len(r['q_p']))
                pool = np.r_[np.delete(r['q_p'], j), r['q_r']]; q = r['q_p'][j]
                vs.append(((pool < q).sum() + 0.5 * (pool == q).sum()) / len(pool))
            vals[key] = vs
        null.append(_participant_mean(vals, part) - 0.5)
    null = np.array(null)
    return dict(best=real, z=float(real / null.std()) if null.std() > 0 else np.nan,
                p_up=float((null >= real).mean()), p_down=float((null <= real).mean()),
                p_two=float((np.abs(null) >= abs(real)).mean()), null_mean=float(null.mean()), null_sd=float(null.std()))
