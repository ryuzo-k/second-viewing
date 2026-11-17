"""Figures for the paper, as plain SVG (black lines, Times). Numbers come from results/*.json."""
import json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(HERE, 'results', '001187.json')))
F = 'font-family="Times New Roman, Times, serif"'


def t(x, y, s, size=13, anchor='middle', extra=''):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" {extra}>{s}</text>'


def task():
    """Figure 1: one encoding trial, the test phase, and which images count as once, twice and new."""
    o = [f'<svg viewBox="0 0 640 250" xmlns="http://www.w3.org/2000/svg" {F} role="img" '
         'aria-label="Task: an encoding trial shows images A, B and C, then probe B; at test, B was seen twice, C once and E is new.">']
    sq = lambda x, y, s, d=False: (f'<rect x="{x}" y="{y}" width="40" height="40" fill="none" stroke="#000" stroke-width="1"'
                                   + (' stroke-dasharray="3 3"' if d else '') + '/>' + t(x + 20, y + 25, s, 15))
    o.append(t(0, 18, 'Encoding phase: 140 working-memory trials', 14, 'start'))
    for i, s in enumerate('ABC'):
        o.append(sq(20 + i * 52, 36, s))
    o.append(t(98, 96, '1 or 3 novel images', 12))
    o.append(f'<line x1="186" y1="56" x2="262" y2="56" stroke="#000" stroke-width="1" stroke-dasharray="2 3"/>')
    o.append(t(224, 48, 'delay', 12))
    o.append(sq(272, 36, 'B'))
    o.append(t(292, 96, 'probe: was it shown?', 12))
    o.append(t(380, 50, 'A probe repeats an image from the same', 12, 'start'))
    o.append(t(380, 66, 'trial or from an earlier one. That image', 12, 'start'))
    o.append(t(380, 82, 'has then been seen twice.', 12, 'start'))
    o.append('<line x1="0" y1="122" x2="640" y2="122" stroke="#000" stroke-width="0.5"/>')
    o.append(t(0, 146, 'Test phase, a median 11 min later: up to 400 images', 14, 'start'))
    for i, (s, lab) in enumerate((('B', 'twice'), ('E', 'new'), ('C', 'once'))):
        o.append(sq(20 + i * 70, 162, s))
        o.append(t(40 + i * 70, 222, lab, 12))
    o.append(t(380, 176, 'Old or new, with confidence 1–6.', 12, 'start'))
    o.append(t(380, 192, 'Images: 200 new, 140 once, 60 twice.', 12, 'start'))
    o.append(t(380, 208, 'Last viewing to test: median 35 min.', 12, 'start'))
    o.append('</svg>')
    return '\n'.join(o)


def forest():
    """Figure 2: k with ±1.96 bootstrap SD, preregistered test and robustness checks (log scale)."""
    rep = R['repetition']['result']; rob = R['robustness']
    rows = [('Preregistered test', rep['log']),
            ('With memory terms', rep['log_memory']),
            ('Probe from the same trial', rob['repetition_probe_same_trial']['result']['log']),
            ('Probe from an earlier trial', rob['repetition_probe_earlier_trial']['result']['log']),
            ('Answers after 1 s', rob['repetition_rt_ge_1s']['result']['log']),
            ('Window 0.2–0.5 s, all answers', rob['repetition_window_0.2_0.5']['result']['log'])]
    lo, hi, x0, x1 = -0.4, 0.4, 250, 610
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    top, dy = 30, 30; n = len(rows) + 1; bottom = top + n * dy
    o = [f'<svg viewBox="0 0 640 {bottom + 60}" xmlns="http://www.w3.org/2000/svg" {F} role="img" '
         'aria-label="Estimates of k with 95% intervals; all include zero and lie above the estimate expected at k = −0.3.">']
    ex = -0.3 * 0.7
    o.append(f'<line x1="{X(0):.1f}" y1="{top - 14}" x2="{X(0):.1f}" y2="{bottom}" stroke="#000" stroke-width="0.6"/>')
    o.append(f'<line x1="{X(ex):.1f}" y1="{top - 14}" x2="{X(ex):.1f}" y2="{bottom}" stroke="#000" stroke-width="0.8" stroke-dasharray="4 3"/>')
    o.append(t(X(ex), top - 20, 'expected if k = −0.3', 12))
    for i, (lab, v) in enumerate(rows):
        y = top + i * dy + 10; sd = v['k'] / v['z']
        o.append(t(0, y + 4, lab, 13, 'start'))
        o.append(f'<line x1="{X(v["k"] - 1.96 * sd):.1f}" y1="{y}" x2="{X(v["k"] + 1.96 * sd):.1f}" y2="{y}" stroke="#000" stroke-width="1.2"/>')
        o.append(f'<circle cx="{X(v["k"]):.1f}" cy="{y}" r="3.5" fill="#000"/>')
    lo1 = rob['repetition_leave_one_out']; y = top + len(rows) * dy + 10
    o.append(t(0, y + 4, 'Leaving out each participant', 13, 'start'))
    o.append(f'<rect x="{X(lo1["min"]):.1f}" y="{y - 4}" width="{X(lo1["max"]) - X(lo1["min"]):.1f}" height="8" fill="#000"/>')
    o.append(f'<line x1="{x0}" y1="{bottom}" x2="{x1}" y2="{bottom}" stroke="#000" stroke-width="0.8"/>')
    for v in (-0.4, -0.2, 0, 0.2, 0.4):
        o.append(f'<line x1="{X(v):.1f}" y1="{bottom}" x2="{X(v):.1f}" y2="{bottom + 5}" stroke="#000" stroke-width="0.8"/>')
        o.append(t(X(v), bottom + 20, ('−' if v < 0 else '') + f'{abs(v):g}', 12))
    o.append(t((x0 + x1) / 2, bottom + 44, 'Estimate of k (log-linear model)', 13))
    o.append('</svg>')
    return '\n'.join(o)


def task_narrow():
    """Figure 1 for narrow screens: the same content, stacked."""
    o = [f'<svg viewBox="0 0 360 330" xmlns="http://www.w3.org/2000/svg" {F} role="img" '
         'aria-label="Task: an encoding trial shows images A, B and C, then probe B; at test, B was seen twice, C once and E is new.">']
    sq = lambda x, y, s: (f'<rect x="{x}" y="{y}" width="36" height="36" fill="none" stroke="#000" stroke-width="1"/>' + t(x + 18, y + 23, s, 15))
    o.append(t(0, 16, 'Encoding phase: 140 working-memory trials', 13, 'start'))
    for i, s in enumerate('ABC'):
        o.append(sq(8 + i * 46, 30, s))
    o.append(f'<line x1="150" y1="48" x2="220" y2="48" stroke="#000" stroke-width="1" stroke-dasharray="2 3"/>')
    o.append(t(185, 40, 'delay', 12)); o.append(sq(230, 30, 'B'))
    o.append(t(80, 86, '1 or 3 novel images', 12)); o.append(t(248, 86, 'probe: was it shown?', 12))
    o.append(t(0, 112, 'A probe repeats an image from the same trial', 12, 'start'))
    o.append(t(0, 128, 'or from an earlier one. That image has then', 12, 'start'))
    o.append(t(0, 144, 'been seen twice.', 12, 'start'))
    o.append('<line x1="0" y1="162" x2="360" y2="162" stroke="#000" stroke-width="0.5"/>')
    o.append(t(0, 184, 'Test phase, a median 11 min later: up to 400 images', 13, 'start'))
    for i, (s, lab) in enumerate((('B', 'twice'), ('E', 'new'), ('C', 'once'))):
        o.append(sq(8 + i * 62, 198, s)); o.append(t(26 + i * 62, 252, lab, 12))
    o.append(t(0, 278, 'Old or new, with confidence 1–6.', 12, 'start'))
    o.append(t(0, 294, 'Images: 200 new, 140 once, 60 twice.', 12, 'start'))
    o.append(t(0, 310, 'Last viewing to test: median 35 min.', 12, 'start'))
    o.append('</svg>')
    return '\n'.join(o)


def forest_narrow():
    """Figure 2 for narrow screens: each label above its line."""
    rep = R['repetition']['result']; rob = R['robustness']
    rows = [('Preregistered test', rep['log']),
            ('With memory terms', rep['log_memory']),
            ('Probe from the same trial', rob['repetition_probe_same_trial']['result']['log']),
            ('Probe from an earlier trial', rob['repetition_probe_earlier_trial']['result']['log']),
            ('Answers after 1 s', rob['repetition_rt_ge_1s']['result']['log']),
            ('Window 0.2–0.5 s, all answers', rob['repetition_window_0.2_0.5']['result']['log'])]
    lo, hi, x0, x1 = -0.4, 0.4, 16, 344
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    top, dy = 36, 40; n = len(rows) + 1; bottom = top + n * dy
    o = [f'<svg viewBox="0 0 360 {bottom + 56}" xmlns="http://www.w3.org/2000/svg" {F} role="img" '
         'aria-label="Estimates of k with conservative intervals; all include zero and lie above the estimate expected at k = −0.3.">']
    ex = -0.3 * 0.7
    o.append(f'<line x1="{X(0):.1f}" y1="{top - 10}" x2="{X(0):.1f}" y2="{bottom}" stroke="#000" stroke-width="0.6"/>')
    o.append(f'<line x1="{X(ex):.1f}" y1="{top - 10}" x2="{X(ex):.1f}" y2="{bottom}" stroke="#000" stroke-width="0.8" stroke-dasharray="4 3"/>')
    o.append(t(X(ex), top - 18, 'expected if k = −0.3', 12))
    for i, (lab, v) in enumerate(rows):
        y = top + i * dy + 26; sd = v['k'] / v['z']
        o.append(f'<rect x="0" y="{y - 22}" width="{len(lab) * 5.9 + 4:.0f}" height="16" fill="#fff"/>')
        o.append(t(0, y - 10, lab, 12, 'start'))
        o.append(f'<line x1="{X(v["k"] - 1.96 * sd):.1f}" y1="{y}" x2="{X(v["k"] + 1.96 * sd):.1f}" y2="{y}" stroke="#000" stroke-width="1.2"/>')
        o.append(f'<circle cx="{X(v["k"]):.1f}" cy="{y}" r="3.5" fill="#000"/>')
    lo1 = rob['repetition_leave_one_out']; y = top + len(rows) * dy + 26
    o.append(f'<rect x="0" y="{y - 22}" width="{len("Leaving out each participant") * 5.9 + 4:.0f}" height="16" fill="#fff"/>')
    o.append(t(0, y - 10, 'Leaving out each participant', 12, 'start'))
    o.append(f'<rect x="{X(lo1["min"]):.1f}" y="{y - 4}" width="{X(lo1["max"]) - X(lo1["min"]):.1f}" height="8" fill="#000"/>')
    o.append(f'<line x1="{x0}" y1="{bottom}" x2="{x1}" y2="{bottom}" stroke="#000" stroke-width="0.8"/>')
    for v in (-0.4, -0.2, 0, 0.2, 0.4):
        o.append(f'<line x1="{X(v):.1f}" y1="{bottom}" x2="{X(v):.1f}" y2="{bottom + 5}" stroke="#000" stroke-width="0.8"/>')
        o.append(t(X(v), bottom + 20, ('−' if v < 0 else '') + f'{abs(v):g}', 12))
    o.append(t((x0 + x1) / 2, bottom + 42, 'Estimate of k (log-linear model)', 13))
    o.append('</svg>')
    return '\n'.join(o)


if __name__ == '__main__':
    out = os.path.join(HERE, 'results')
    open(os.path.join(out, 'figure1.svg'), 'w').write(task())
    open(os.path.join(out, 'figure2.svg'), 'w').write(forest())
    open(os.path.join(out, 'figure1_narrow.svg'), 'w').write(task_narrow())
    open(os.path.join(out, 'figure2_narrow.svg'), 'w').write(forest_narrow())
    print('ok')
