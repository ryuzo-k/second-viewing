"""Calibration of the repetition test and the pair analysis on synthetic worlds built on the real 001187 task (images,
answers, timing, neurons and their noise correlations), where the truth is known. Prints one JSON line per world.

Usage: DATA=... python calibrate.py MODE [--workers N] WORLD:SEED [WORLD:SEED ...]
  MODE repetition                k and its bootstrap z (log, identity, log with memory terms; 100 crossed bootstrap draws)
       repetition-label-shuffle  the same z and the label-shuffle test (200 shuffles of the twice/once labels)
       pairs                     pair statistic and its swap test (500 null draws)
  WORLD  world names joined by '+', each optionally with a size (e.g. self=-0.2+corr=1.0); 'none' adds nothing.
         Worlds are described in simulate.simulate_session; sizes default to the values below. Each output line
         records the world with all its sizes, so `python calibrate.py MODE 'WORLD:SEED'` reproduces it.
In the repetition modes the synthetic tuned neurons of each session are cut at random to the number of tuned neurons in
the real session, so the analysis sees as many neurons as in the real data."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')   # one BLAS thread per world
import argparse, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import analysis as A
import load_001187
from simulate import simulate_session

REPETITION_SIZES = dict(sme=0.3, state=0.3, recall=0.15, replay=0.3, corr=1.0, novel_add=0.3, self=-0.2, self_step=-0.15,
                        gain=-0.1, probe_typ_both=-0.2, meaning=0.1, drift=-0.3)
PAIRS_SIZES = dict(sme=0.3, recall=0.15, state=0.3, pair=0.25, novel_add=0.3, replay=0.3, corr=1.0, pool=-0.2, meaning=0.1)


def parse(world, defaults):
    sizes = dict(defaults)
    for part in world.split('+'):
        if '=' in part: k, v = part.split('='); sizes[k] = float(v)
    names = [x.split('=')[0] for x in world.split('+') if x.split('=')[0] != 'none']
    return {k: sizes[k] for k in names}


def build(params, rng, match):
    """Synthetic sessions of every 001187 session. 'pair' plants A->B pairs (40% of neurons as B, each with one A at
    another location); 'pool' acts on the new-image pool (IDs >= 1001)."""
    pf = {}; out = []
    for p in load_001187.files():
        s = load_001187.load_session(p); n_real = len(A.tuned_neurons(s)) if match else None
        truth = {k: v for k, v in params.items() if k != 'pair'}
        if 'pool' in params: truth['pool_ids'] = set(int(i) for i in s['test']['id'] if i >= 1001)
        if 'pair' in params:
            nU = s['enc'].shape[0]; planted = []
            for B in rng.permutation(nU)[: max(1, int(0.4 * nU))]:
                cand = [a for a in range(nU) if s['area'][a] != s['area'][B]]
                if cand: planted.append((int(rng.choice(cand)), int(B)))
            truth.update(pair=params['pair'], pairs=planted)
        ss = simulate_session(s, truth, rng, pf)
        if match:
            tu = A.tuned_neurons(ss)
            if len(tu) > n_real: tu = sorted(rng.choice(tu, n_real, replace=False).tolist())
            ss['tuned'] = tu
        out.append(ss)
    return out


def repetition_fits(sessions, rng):
    return {name: A.boot_repetition(sessions, rng, n=100, scale=scale, memory=memory)
            for name, scale, memory in (('log', 'log', False), ('identity', 'identity', False), ('log_memory', 'log', True))}


def run(mode, job):
    A.category = load_001187.cat_of
    world, seed = job.rsplit(':', 1); seed = int(seed)
    params = parse(world, PAIRS_SIZES if mode == 'pairs' else REPETITION_SIZES)
    rng = np.random.default_rng(seed)
    sessions = build(params, rng, match=mode != 'pairs')
    res = dict(world='+'.join(f'{k}={v}' for k, v in params.items()) or 'none', params=params, seed=seed)
    if mode == 'pairs':
        out = A.pairs(sessions); st = A.pair_stats(out)
        res['swap_test'] = {k: round(v, 4) for k, v in A.pair_swap_test(out, rng, n=500).items()}
        res['pairs'] = dict(best=round(float(st['best']), 3), rank=round(float(st['rank']), 3),
                            n_real=int(sum(len(r['u']) for v in out.values() for r in v['res'])))
    elif mode == 'repetition':
        res['n_tuned'] = int(sum(len(s['tuned']) for s in sessions))
        for name, (e, sd) in repetition_fits(sessions, rng).items():
            res[name] = dict(twice=round(float(e['twice']), 4), k=round(float(e['k']), 4),
                             z_twice=round(float(e['twice'] / sd['twice']), 4), z_k=round(float(e['k'] / sd['k']), 4),
                             k_first_viewing=round(float(e['k_first_viewing']), 4))
    else:
        for name, (e, sd) in repetition_fits(sessions, rng).items():
            res[name] = dict(k=round(float(e['k']), 4), z=round(float(e['k'] / sd['k']), 4))
        res['label_shuffle'] = A.label_shuffle(sessions, n=200, seed=seed + 1)
    return json.dumps(res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['repetition', 'repetition-label-shuffle', 'pairs'])
    ap.add_argument('jobs', nargs='+', metavar='WORLD:SEED')
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    if a.workers == 1:
        for j in a.jobs: print(run(a.mode, j), flush=True)
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            for line in ex.map(run, [a.mode] * len(a.jobs), a.jobs): print(line, flush=True)
