"""Exploratory, not preregistered: the uniform effect of the extra viewing at test (test x twice-seen, at x = 0), from
the preregistered model and bootstrap, with the same seed as run.py. Also returns k, which must equal the registered
result."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analysis as A, run

sessions, _ = run.prepare_001187()
rng = np.random.default_rng(20261120)
e, sd = A.boot_repetition(sessions, rng, n=100, scale='log', memory=False)
R = A.all_repetition_rows(sessions); x = R[R[:, 2] >= 1, 3]
out = dict(twice=float(e['twice']), twice_sd=float(sd['twice']), k=float(e['k']), k_sd=float(sd['k']),
           x_mean_old_rows=float(x.mean()))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(run.RES, 'extra_uniform_effect.json'), 'w'), indent=1)
