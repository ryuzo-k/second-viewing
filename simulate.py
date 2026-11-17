"""Synthetic worlds for calibration and dry runs: the real task (images, answers, timing, neurons and their noise
correlations) with synthetic spike counts drawn from a known truth."""
import numpy as np
import analysis as A


def simulate_session(sess, truth, rng, image_factor):
    """Copy of sess with synthetic encoding-phase counts and synthetic test counts ('test_count', in TEST_WINDOW).
    image_factor: dict (image set, image ID) -> shared salience (the same image has the same value in every session).
    truth: world name -> size. Worlds with no effect (nuisances the analyses must not mistake for one):
      corr       neurons of one participant share noise as in the real encoding phase (size = share of the real correlation)
      sme        first-viewing responses are stronger to images that are later remembered
      state      an encoding-phase trial state shared by all neurons and linked to later memory; graded decision signal
      recall     hits only: category reinstatement x (1 + size * x) and decision neurons x 1.2
      replay     hits only: each neuron re-expresses its own first-viewing response to that image
      gain       every old image x (1 + size) per showing
      novel_add  an additive (Hz) novelty signal on new images in 20% of neurons
      pool       new images in pool_ids drive neurons with a high x less
      meaning    the participant's own meaning of each image raises every neuron's response at both viewings
      probe_typ_both  twice-shown images drive neurons differently by x, at every viewing (a property of the images)
      drift      responses to the preferred category change across the encoding phase (not carried to the test)
    Worlds with an effect: self (repetition: each showing scales the response by 1 + size * x, so k ≈ size), self_step
    (1 + size per showing, only where x > 0), pair (pairs: Hebbian change on planted A->B pairs, given in truth['pairs'])."""
    s = dict(sess); T = sess['test']; nU = sess['enc'].shape[0]
    real_rate = np.array([[sess['enc'][u, sess['enc_cat'] == c].mean() if (sess['enc_cat'] == c).any() else 0 for c in range(1, 6)] for u in range(nU)])
    lam = real_rate + 0.05
    fano = 1.4
    ids = np.unique(np.concatenate([sess['enc_id'], T['id']]))
    sh = np.array([image_factor.setdefault((sess['image_set'], i), rng.normal()) for i in ids])
    noise = rng.normal(size=(nU, len(ids)))
    if truth.get('corr'):
        if '_res' not in sess: A.set_pop(sess); A.state_residuals(sess)
        Rr = sess['_res'][sess['_good']]
        if len(Rr) >= 2:
            Cm = np.corrcoef(Rr); Cm = np.nan_to_num(Cm) * truth['corr'] + np.eye(len(Rr)) * (1 - truth['corr'])
            Lc = np.linalg.cholesky(Cm + 1e-6 * np.eye(len(Rr))); gi_ = np.where(sess['_good'])[0]
            noise[gi_] = Lc @ noise[gi_]
    z = 0.35 * sh[None, :] + np.sqrt(1 - 0.35 ** 2) * noise
    col = {i: k for k, i in enumerate(ids)}
    old = (T['shown'] == 1) & np.isin(T['id'], sess['enc_id'])
    so_of = dict(zip(T['id'][old], T['said_old'][old] - T['said_old'][old].mean()))
    if truth.get('sme'):
        for i, d in so_of.items(): z[:, col[i]] += truth['sme'] * d
    ec = sess['enc_cat']
    lam_enc = np.where(ec[None, :] > 0, lam[:, np.maximum(ec, 1) - 1], lam.mean(1, keepdims=True))
    zc = z[:, [col[i] for i in sess['enc_id']]]
    tune_enc = lam_enc * np.exp(0.4 * zc - 0.08)
    mean_ing = rng.normal(size=len(ids)) if truth.get('meaning') else np.zeros(len(ids))
    if truth.get('meaning'):
        tune_enc = tune_enc * np.exp(truth['meaning'] * mean_ing[[col[i] for i in sess['enc_id']]])[None, :]
    if truth.get('probe_typ_both'):
        tw_ids = set(T['id'][T['shown'] == 2])
        relp = lam / lam.mean(1, keepdims=True) - 1
        for j_, i_ in enumerate(sess['enc_id']):
            if i_ in tw_ids and A.category(i_) > 0: tune_enc[:, j_] *= np.clip(1 + truth['probe_typ_both'] * relp[:, A.category(i_) - 1], 0.05, None)
    if truth.get('drift'):
        relp = lam / lam.mean(1, keepdims=True) - 1
        for j_, c_ in enumerate(sess['enc_cat']):
            if c_ > 0: tune_enc[:, j_] *= np.exp(truth['drift'] * sess['enc_t'][j_] * relp[:, c_ - 1])
    if truth.get('state'):
        stt = np.exp(rng.normal(0, 0.2, len(sess['enc_id'])) + np.array([truth['state'] * so_of.get(i, 0.0) for i in sess['enc_id']]))
        tune_enc = tune_enc * stt[None, :]

    def nb(mu):
        mu = np.maximum(mu, 1e-3); r = mu / (fano - 1); return rng.negative_binomial(r, r / (r + mu))
    enc = nb(tune_enc).astype(float); s['enc'] = enc
    s['enc_s'] = rng.binomial(enc.astype(int), 0.55).astype(float)   # 0.2-0.7 s counts: a part of the 0.2-1.2 s counts
    A.set_pop(s)
    if 'pre' in sess:                                            # before the image: baseline rate x trial state only
        lam_b = sess['pre'].mean(1) + 0.05; Wp = A.PRE[1] - A.PRE[0]
        mu_b = np.repeat(lam_b[:, None], len(sess['enc_id']), 1)
        if truth.get('state'): mu_b = mu_b * stt[None, :]
        s['pre'] = nb(mu_b * Wp) / Wp
    if 'hold' in sess:                                          # hold period (001187): some neurons keep firing for the image
        hm = sess['has_hold']; Wm = A.HOLD[1] - A.HOLD[0]; mreal = sess['hold'][:, hm]
        lam_m = (np.nanmean(mreal, 1) if hm.any() else np.zeros(nU)) + 0.05
        rho = np.where(rng.random(nU) < 0.3, 0.7, 0.1)           # persistence of the image-specific part
        mu_m = lam_m[:, None] * np.exp(0.4 * rho[:, None] * zc - 0.08 * rho[:, None] ** 2)
        if truth.get('state'): mu_m = mu_m * stt[None, :]
        mt = np.full(enc.shape, np.nan); mt[:, hm] = nb(mu_m[:, hm] * Wm) / Wm
        s['hold'] = mt
    first = {}
    for j, i in enumerate(sess['enc_id']): first.setdefault(i, enc[:, j])
    drift = np.exp(rng.normal(0, 0.15, nU))
    ms = rng.random(nU) < 0.15                                   # decision neurons
    nov = rng.random(nU) < 0.2                                   # neurons that add spikes to new images (novel_add)
    mean_first = enc.mean(1) + 1e-6
    rel_all = lam / lam.mean(1, keepdims=True) - 1
    W = A.TEST_WINDOW[1] - A.TEST_WINDOW[0]
    tc = np.zeros((nU, len(T['id'])))
    for t, i in enumerate(T['id']):
        c = A.category(i)
        if c == 0: continue
        mu = lam[:, c - 1] * np.exp(0.4 * z[:, col[i]] - 0.08) * drift
        if truth.get('meaning'): mu = mu * np.exp(truth['meaning'] * mean_ing[col[i]])
        conf = T['conf'][t]; so = T['said_old'][t]
        if truth.get('state'): mu = mu * np.where(ms, np.clip(1 + 0.12 * (conf - 3.5), 0.5, None), 1.0)
        else: mu = mu * np.where(ms & (so == 1), 1.3, 1.0)
        if truth.get('novel_add') and T['shown'][t] == 0:
            mu = mu + np.where(nov, truth['novel_add'] * lam.mean(1), 0.0)
        if truth.get('pool') and T['shown'][t] == 0 and i in truth.get('pool_ids', ()):
            mu = mu * np.clip(1 + truth['pool'] * rel_all[:, c - 1], 0.05, None)
        if T['shown'][t] in (1, 2) and i in first:
            nexp = int(T['shown'][t]); rel = rel_all[:, c - 1]          # gain, self, self_step act once per showing
            if truth.get('replay') and so == 1:
                mu = mu * np.exp(truth['replay'] * (0.4 * z[:, col[i]]))
            if truth.get('gain'): mu = mu * (1 + truth['gain']) ** nexp
            if truth.get('probe_typ_both') and nexp == 2: mu = mu * np.clip(1 + truth['probe_typ_both'] * rel, 0.05, None)
            if truth.get('recall') and so == 1: mu = mu * np.clip(1 + truth['recall'] * rel, 0.05, None) * np.where(ms, 1.2, 1.0)
            if truth.get('self'): mu = mu * np.clip(1 + truth['self'] * rel, 0.05, None) ** nexp
            if truth.get('self_step'): mu = mu * np.where(rel > 0, 1 + truth['self_step'], 1.0) ** nexp
            if truth.get('pair'):
                a = first[i] / mean_first
                for (Ai, Bi) in truth['pairs']: mu[Bi] += truth['pair'] * a[Ai] * a[Bi] * lam[Bi].mean()
        tc[:, t] = nb(mu * W)
    s['test_count'] = tc
    return s
