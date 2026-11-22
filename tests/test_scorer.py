"""The batched scorer gives exactly wakeforge's keyword_score, for every keyword."""
import numpy as np
from ovos_ww_plugin_wakeforge.zeroshot import keyword_score

from ovos_audio_transformer_plugin_overheard_manners.spotter import BatchScorer


def test_batch_scorer_equals_wakeforge():
    rng = np.random.default_rng(1)
    keywords = [list(rng.integers(4, 392, size=n)) for n in (2, 3, 5, 8, 12, 17)] + [[10, 10, 11], [7]]
    score = BatchScorer(keywords)
    for _ in range(5):
        p = rng.dirichlet(np.full(392, 0.05), size=75)
        lp = np.log(np.clip(p, 1e-10, None))
        ref = np.array([keyword_score(lp, k) for k in keywords])
        assert np.array_equal(score(lp), ref)
