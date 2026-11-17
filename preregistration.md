# Does seeing an image once more change a human medial temporal lobe neuron's later response in proportion to its category tuning?

## Background
Responses of human medial temporal lobe (MTL) neurons to an image decline when the image is repeated within a session (Pedreira et al., 2010, J Neurophysiol 103:97). We ask whether a single extra viewing of a novel image leaves a trace about half an hour later, and whether its size scales with how strongly the neuron responds to the image's category.

## Task and data
DANDI 001187, version 0.240904.2347 (Daume, Cheng & Rutishauser, 2024, https://doi.org/10.48324/dandi.001187/0.240904.2347): 46 sessions, 39 participants, single neurons in hippocampus and amygdala.
- Encoding phase: 140 working-memory trials. Each shows 1 or 3 novel images from 5 categories, then a probe image; the participant answers whether the probe was in the trial. Probes are images from the same trial or from an earlier one, so some images are seen twice.
- Test phase, a median 11 min (range 2–35) after the encoding phase: 400 images in a complete session (200 new, 140 seen once, 60 seen twice); the participant answers old or new with confidence 1–6 (4–6 = old). An image's test comes a median 35 min (5th–95th percentile 15–63) after its last viewing.
- Which images are probed is fixed by the task design and identical for everyone with the same image set (3 sets).

## Hypothesis
For each neuron and category, x = the neuron's mean response to that category's encoding-phase images that never appear at test, divided by its mean response to all such images, minus 1 (0.2–1.2 s after onset; over tuned neurons × categories, median −0.08, interquartile range −0.32 to 0.21).
Model: change in log test response per extra viewing = k · x.
Prediction: k < 0 (more decline where the neuron responds more). At k = −0.2, one extra viewing lowers the response by 4% at x = 0.21 and raises it by 7% at x = −0.32 (the quartiles of x).

## Blinding
We have examined all encoding-phase spikes, test-phase behaviour and images. No test-phase spike has been counted. The code reads test-phase spikes only when OSF_REGISTRATION_URL.txt holds the osf.io registration URL and UNBLIND=after-registration is set.
Already examined: at first viewing, twice- vs once-seen images differ by +0.036 in the coefficient on x (SD 0.038 over 40 resamples of participants; +0.027, SD 0.045, with the context terms of step 3). Twice-seen images were first shown earlier in the phase and more often in one-image trials. Step 3 adjusts for both.

## Analysis
Code: github.com/ryuzo-k/second-viewing, at the commit cited in this registration. Where text and code differ, the code governs.
1. Neurons are excluded if, between phases, waveform correlation is < 0.98, amplitude ratio is outside 0.86–1.12, or firing-rate ratio is outside 0.33–3.3; or if either phase has > 1% of inter-spike intervals under 3 ms or < 40 spikes. These use whole-phase spikes, not image or answer labels.
2. Tuned neurons: ≥ 20 spikes in the 0.2–1.2 s windows, ≥ 3 never-tested images per category, and category responses that differ on them (one-way ANOVA, p < 0.05).
3. For each tuned neuron and each image seen once or twice, spikes 0.2–0.7 s after onset are counted at first viewing and at test. Test trials answered before 0.7 s are excluded, so the window never contains the answer. Poisson regression with one effect per neuron × category × phase.
   - Tested term: test × twice-seen × x (= k).
   - Twice-seen, twice-seen × x and twice-seen × category enter at both phases, so differences present at first viewing are subtracted.
   - First-viewing rows: time in phase, one-image trial, first image of a trial, each also × x.
   - Test rows: position (cubic), whether the previous test image was old, whether it was called old, image typicality (similarity of its CLIP embedding to the other test images of its category and image set, z-scored within category), log time since last viewing, log reaction time; position, typicality, time and reaction time also × x.
   - Answer signal: each neuron's log ratio of responses to new images called old vs new, shrunk toward the population mean by a split-half estimate of its noise, plus a pooled effect of confidence within each answer, is subtracted from every test row.
   - z = k / SD of 100 bootstrap draws resampling participants and images independently.

## Decision
Found if |z| > 1.6, and k has the same sign when (a) the model is fitted on the difference scale (least squares on counts divided by their mean within neuron × category × phase) and (b) said-old × x and confidence 5–6 × x are added. Either sign is reported; the prediction is negative. If any of the three z cannot be computed, the test is reported as not done.

## Calibration
Synthetic datasets built on the real task, answers and noise correlations, with at most each session's number of tuned neurons (131–160 in total; 166 in the real data before exclusion), without exclusion (calibration/ in the code).
- k = 0: decision met in 1 of 310 datasets (0.3%; 95% CI 0–1.8%), including datasets whose tuning drifts across the encoding phase and datasets with person-specific image meaning.
- k < 0: found at k = −0.1 in 3 of 15 datasets, −0.2 in 17 of 30, −0.3 in 28 of 30, −0.5 in 25 of 25; a 15% decline only in categories above the neuron's mean, 9 of 15. Estimates are about 70% of the true k, because x is measured with noise.
- The threshold |z| > 1.6 was fixed before these datasets were made. Under k = 0, z had SD 0.60: resampling participants and images independently overestimates variance (Owen, 2007, Ann Appl Stat 1:386).

## Limits
Adaptation and learning are not separated. The second viewing included a working-memory judgement. If nothing is found, we conclude only that k ≤ −0.3 would probably have been detected.

## Reported in every case
k with each participant left out; with twice/once labels shuffled within image set and category; for probes from the same vs an earlier trial; for answers after 1 s; with 0.2–0.5 s windows and fast answers kept; trials excluded per image group.

## Exploratory
Neuron pairs, no claim on their own: whether neuron B's test response depends on the product of its own and a same-participant neuron A's first-viewing responses, compared with neurons of other participants who saw the same images (|z| > 2.5 in 7 of 280 synthetic datasets with no effect, 2.5%).
- Tested in DANDI 000004, version 0.220126.1852 (Chandravadia et al., 2022, https://doi.org/10.48324/dandi.000004/0.220126.1852; 87 sessions, 59 participants), only if |z| > 2.5 here and two controls (spikes before image onset; images swapped within category) give |z| ≤ 2.5. Confirmed there if the sign matches and the one-sided z exceeds 1.645.
- Already examined: a pilot on encoding-phase probes, its form fixed beforehand, gave z = −0.25 (probes from the same trial) and +0.15 (from an earlier trial); five variants then tried on the latter all gave |z| < 2 (not in the code).
- Also reported: the same analysis on the hold period after the last image of a trial, and per brain area.
