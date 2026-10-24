"""One screen observation in, the move out."""
from __future__ import annotations

import math

import numpy as np

from .encode import INPUTS, QUANTILES, VALUE_SHIFT, encode
from .feat import Featurizer
from .master import Master


class Policy:
    def __init__(self, model, master, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        self.f = Featurizer(master if isinstance(master, Master) else Master(master))

    def decide(self, obs):
        x = encode(obs, self.f)
        logit, _adv, value = self.sess.run(["logit", "adv", "value"], {k: x[k] for k in INPUTS})
        n = len(obs["legal"])
        z = logit[0, :n] - logit[0, :n].max()
        probs = np.exp(z) / np.exp(z).sum()
        k = int(np.argmax(probs))
        return {"action": list(obs["legal"][k]), "probs": [float(p) for p in probs],
                "score_quantiles": {f"{int(q * 100)}%": int(math.exp(float(v) + VALUE_SHIFT))
                                    for q, v in zip(QUANTILES, value[0])}}
