"""Stage-1 hits per hour of ordinary talk and recall, per sensitivity offset, from calib_plugin.py's output.

    python calib_report.py en da de es
"""
import json
import sys

import numpy as np

DEBOUNCE = 25            # blocks (2 s), as the plugin's spotter


def held(m):
    """confirm_blocks=2: what two windows in a row both reach."""
    return np.minimum(m[1:], m[:-1])


def main(langs):
    for lang in langs:
        d = np.load(f"scores/calibp_{lang}.npz", allow_pickle=True)
        neg, utt = d["neg"], d["utt"][1:]
        hours = len(neg) * 0.08 / 3600
        meta = [json.loads(m) for m in d["meta"]]
        print(f"== {lang} ({hours:.2f} h of FLEURS)")
        print(" offset  stage-1 hits/h   recall  clean  room  noise  babble")
        hn = held(neg)
        for off in np.round(np.arange(0.0, 0.125, 0.01), 2):
            idx = np.flatnonzero(((hn - off) >= 0).any(1))
            n, last, lu = 0, -10 ** 9, -1
            for i in idx:
                if utt[i] != lu or i - last > DEBOUNCE:
                    n += 1; last, lu = i, utt[i]
            rec, by = [], {}
            for mt, m in zip(meta, d["pos"]):
                hit = bool(((held(np.asarray(m, np.float32))[:, mt["idx"]] - off) >= 0).any())
                rec.append(hit); by.setdefault(mt["cond"], []).append(hit)
            print(f" {off:5.2f}   {n / hours:8.1f}       {np.mean(rec):.2f}   "
                  + "  ".join(f"{np.mean(by[c]):.2f}" for c in ["clean", "room", "noise10", "babble10"]))


if __name__ == "__main__":
    main(sys.argv[1:])
