"""Stage 1: a sensitive first ear for swearing in the room.

The phrases of the device's language and the English base phrases are spotted
zero-shot from their phones with WakePhoneHuBERT, using the scoring of
ovos-ww-plugin-wakeforge's zero-shot wake word engine (the same model, the same
CTC keyword score, the same thresholds rule). It is set far more sensitive than a
wake word: a hit here only means "worth a closer look", and stage 2 (local STT and
the word lists) decides. A phrase fires when one of its pronunciations holds its
threshold for ``confirm_blocks`` windows in a row, as in the wakeforge engine.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

LOCALE = Path(__file__).parent / "locale"

#: Thresholds are wakeforge's default_threshold(n_phones) plus this offset;
#: higher is stricter. With ~15 phrases of up to 10 pronunciations each, the
#: default rule alone lets the first ear fire some 300-600 times an hour of
#: ordinary talk. 0.07 brings that to about 45 an hour in English and 85 in
#: Danish (1 h of FLEURS each), so stage 2's STT runs about once a minute while
#: people talk, and catches about half the English and a third of the Danish
#: test phrases. Measured with evaluation/calib_plugin.py, see its README.
DEFAULT_OFFSET = 0.07


@dataclass
class Phrase:
    text: str
    lang: str          # the language the phrase belongs to ("en" for the base list)
    ipa: List[str]     # pronunciations, phones separated by spaces


def lang_code(lang: str) -> str:
    """'da-DK' -> 'da'."""
    return (lang or "en").split("-")[0].split("_")[0].lower()


def locale_dir(lang: str) -> Optional[Path]:
    """The locale folder for a language ('da', 'da-DK' or 'da_dk' -> locale/da-DK)."""
    code = lang_code(lang)
    for d in sorted(LOCALE.iterdir()):
        if d.is_dir() and lang_code(d.name) == code:
            return d
    return None


def load_phrases(lang: str) -> List[Phrase]:
    """The phrases listened for with the device in ``lang``: its own and the English base list,
    the English ones with both English and local pronunciations."""
    code = lang_code(lang)
    phrases = []
    for src in dict.fromkeys([code, "en"]):
        folder = locale_dir(src)
        f = folder / "overheard.json" if folder else None
        if not f or not f.exists():
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        for p in data["phrases"]:
            ipa = list(p["ipa"])
            if src == "en" and code != "en":
                ipa += p.get("ipa_local", {}).get(code, [])
            phrases.append(Phrase(p["text"], src, list(dict.fromkeys(ipa))))
    return phrases


class BatchScorer:
    """wakeforge's keyword_score for many keywords at once.

    Same scorer, same numbers and tie-breaking (tests/test_scorer.py checks it
    against wakeforge's own), but all pronunciations move through the window's
    frames together. With ~90 pronunciations wakeforge's per-keyword loop takes
    about 100 ms per 80 ms block; this takes a few.
    """

    NEG = -1e30

    def __init__(self, keywords: List[List[int]]):
        K = len(keywords)
        S = max(2 * len(k) + 1 for k in keywords)
        self.labels = np.zeros((K, S), dtype=np.int64)
        self.valid = np.zeros((K, S), dtype=bool)
        self.skip = np.zeros((K, S), dtype=bool)
        self.last = np.zeros(K, dtype=np.int64)
        for i, ids in enumerate(keywords):
            n = 2 * len(ids) + 1
            lab = np.zeros(n, dtype=np.int64)
            lab[1::2] = ids
            self.labels[i, :n] = lab
            self.valid[i, :n] = True
            self.skip[i, 2:n] = (lab[2:] != 0) & (lab[2:] != lab[:-2])
            self.last[i] = n - 1
        self.rows = np.arange(K)

    def __call__(self, log_post: np.ndarray) -> np.ndarray:
        NEG = self.NEG
        lp = np.asarray(log_post, dtype=np.float64)
        emit = lp[:, self.labels] - lp.max(axis=1)[:, None, None]          # [T,K,S]
        emit = np.where(self.valid[None], emit, NEG)
        K, S = self.labels.shape
        a = np.full((K, S), NEG)
        origin = np.zeros((K, S), dtype=np.int64)
        best = np.full(K, NEG)
        start = np.full((K, S), NEG)
        start[:, :2] = 0.0
        for t in range(lp.shape[0]):
            prev, src = a.copy(), origin.copy()
            for shift, allowed in ((1, None), (2, self.skip)):
                cand = np.full((K, S), NEG)
                cand[:, shift:] = a[:, :-shift]
                if allowed is not None:
                    cand = np.where(allowed, cand, NEG)
                csrc = np.zeros((K, S), dtype=np.int64)
                csrc[:, shift:] = origin[:, :-shift]
                better = cand > prev
                prev, src = np.where(better, cand, prev), np.where(better, csrc, src)
            better = start > prev
            prev, src = np.where(better, start, prev), np.where(better, t, src)
            a = np.where(self.valid, prev + emit[t], NEG)
            origin = src
            for s in (self.last, self.last - 1):
                best = np.maximum(best, a[self.rows, s] / (t + 1 - origin[self.rows, s]))
        return best


class PhraseSpotter:
    """Scores 80 ms blocks of 16 kHz float audio; returns the phrase that fired, if any."""

    def __init__(self, phrases: List[Phrase], offset: float = DEFAULT_OFFSET,
                 confirm_blocks: int = 2, onnx_threads: int = 1, engine=None, debounce_sec: float = 2.0):
        from ovos_ww_plugin_wakeforge.zeroshot import BLOCK, WINDOW, default_threshold
        self.block, self.window_size = BLOCK, WINDOW
        if engine is None:
            from ovos_ww_plugin_wakeforge.zeroshot import WakePhoneHuBERTZeroShotPlugin
            ipa = [i for p in phrases for i in p.ipa]
            engine = WakePhoneHuBERTZeroShotPlugin(
                "overheard", {"ipa": ipa, "confirm_blocks": confirm_blocks, "onnx_threads": onnx_threads})
        self.engine = engine
        self.score = BatchScorer(engine.keywords)
        # which phrase each pronunciation belongs to
        self.owner: List[Phrase] = [p for p in phrases for _ in p.ipa]
        self.thresholds = np.array([default_threshold(len(k)) + offset for k in engine.keywords])
        self.confirm_blocks = confirm_blocks
        self.debounce_blocks = int(debounce_sec * 16000 / self.block)
        self.reset()

    def score_window(self, audio: np.ndarray) -> np.ndarray:
        """Score of every pronunciation over one 1.5 s window, as the wakeforge engine scores it."""
        from ovos_ww_plugin_wakeforge.zeroshot import featurize, log_posteriors
        feats = featurize(self.engine.session, audio[np.newaxis, :])
        return self.score(log_posteriors(feats[0, :, self.engine.ipa_slice]))

    def reset(self):
        self._window = np.zeros(self.window_size, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._recent: List[np.ndarray] = []
        self._quiet = 0

    def feed(self, audio: np.ndarray) -> Optional[Phrase]:
        """Feed float audio of any length; returns a phrase when one fired in it."""
        self._pending = np.concatenate([self._pending, audio.astype(np.float32)])
        fired = None
        while self._pending.size >= self.block:
            block, self._pending = self._pending[:self.block], self._pending[self.block:]
            self._window = np.concatenate([self._window[self.block:], block])
            scores = self.score_window(self._window)
            self._recent = (self._recent + [scores])[-self.confirm_blocks:]
            if self._quiet:
                self._quiet -= 1
                continue
            if fired is None and len(self._recent) == self.confirm_blocks:
                held = np.min(np.stack(self._recent), axis=0) - self.thresholds
                if (held >= 0).any():
                    fired = self.owner[int(np.argmax(held))]
                    self._recent = []
                    self._quiet = self.debounce_blocks       # one hit per utterance, as the wakeforge engine
        return fired

