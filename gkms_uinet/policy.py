"""One screen observation in, the move out."""
from __future__ import annotations

import math

import numpy as np

from .encode import QUANTILES, VALUE_SHIFT, encode
from .feat import Featurizer
from .master import Master


class Policy:
    """MODEL decides which observation it reads: a version 2 model (it has the input ``t_rel``) takes UIObs v2,
    a version 1 model UIObs v1."""

    def __init__(self, model, master, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.sess.get_inputs()]
        master = master if isinstance(master, Master) else Master(master)
        if "t_rel" in self.inputs:
            from .encode2 import encode2
            from .feat2 import Featurizer2
            self.version, self.f, self.encode = 2, Featurizer2(master), encode2
        else:
            self.version, self.f, self.encode = 1, Featurizer(master), encode

    def decide(self, obs):
        x = self.encode(obs, self.f)
        logit, _adv, value = self.sess.run(["logit", "adv", "value"], {k: x[k] for k in self.inputs})
        n = len(obs["legal"])
        z = logit[0, :n] - logit[0, :n].max()
        probs = np.exp(z) / np.exp(z).sum()
        k = int(np.argmax(probs))
        return {"action": list(obs["legal"][k]), "probs": [float(p) for p in probs],
                "score_quantiles": {f"{int(q * 100)}%": int(math.exp(float(v) + VALUE_SHIFT))
                                    for q, v in zip(QUANTILES, value[0])}}
