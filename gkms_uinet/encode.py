"""A screen observation (UIObs v1) as the network's input arrays, a batch of one."""
from __future__ import annotations

import math

import numpy as np

from .feat import SLOT_CAT, SLOT_NUM, SLOTS, THING_CAT, THING_NUM, slog
from .obs import INF

KIND_HAND, KIND_DECK, KIND_GRAVE, KIND_HOLD, KIND_LOST, KIND_PLAYED, KIND_ENCH, KIND_DRINK = range(1, 9)
ZONES = (("deck", KIND_DECK), ("grave", KIND_GRAVE), ("hold", KIND_HOLD), ("lost", KIND_LOST),
         ("played", KIND_PLAYED))
T_EXT = 6
G_NUM = 47
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
VALUE_SHIFT = 13.0
INPUTS = ("g_num", "g_cat", "st_kind", "st_num", "t_kind", "t_cat", "t_num", "s_cat", "s_num", "t_ext", "a_tok")
# kinds whose lasting layers the training data counted as 20+ turns rather than "no limit"
_LONG_WHEN_UNLIMITED = {"lesson_parameter_multiple"}


def _onehot3(t):
    v = [0.0, 0.0, 0.0]
    if t in (1, 2, 3):
        v[t - 1] = 1.0
    return v


def global_features(o):
    ex, nm = o["exam"], o["num"]
    turn, limit = int(ex["turn"]), max(1, int(ex["limit"]))
    types = list(ex.get("types") or [])
    cur = types[turn - 1] if 0 < turn <= len(types) else 0
    ahead = []
    for k in (1, 2, 3):
        t = types[turn - 1 + k] if turn - 1 + k < len(types) else 0
        ahead += _onehot3(t)
    rest = types[turn - 1:] if turn - 1 < len(types) else []
    counts = [sum(1 for t in rest if t == k) / 12.0 for k in (1, 2, 3)]
    permil = [max(0, int(p)) for p in ex.get("permil") or [0, 0, 0]]
    pmax = max(permil) or 1
    crit = [max(0, int(c)) for c in ex.get("crit") or [0, 0, 0]]
    csum = sum(crit) or 1
    ms = max(1, int(nm.get("max_stamina") or 1))
    cur_permil = permil[cur - 1] if cur in (1, 2, 3) else 0
    stance = o.get("stance") or [None, 0]
    zones = [len(o.get(z) or ()) / 20.0 for z in ("hand", "deck", "grave", "hold", "lost", "played")]
    g = ([math.log1p(max(0, int(nm["score"]))) / 15.0, int(nm["stamina"]) / ms, ms / 60.0, slog(nm.get("block")),
          min(4, int(nm.get("playable") or 0)) / 4.0, min(6, int(nm.get("plays") or 0)) / 6.0,
          turn / limit, (limit - turn) / 12.0, limit / 12.0, max(0, limit - len(types)) / 3.0]
         + _onehot3(cur) + ahead + counts
         + [slog(p) for p in permil] + [p / pmax for p in permil] + [math.log1p(cur_permil) / 12.0]
         + [c / csum for c in crit] + [slog(sum(crit))]
         + zones + [min(3, int(stance[1] or 0)) / 3.0, len(o.get("drinks") or ()) / 4.0,
                    len(o.get("ench") or ()) / 20.0, len(o.get("status") or ()) / 12.0,
                    float(int(nm.get("playable") or 0) > 0)])
    g += [0.0] * (G_NUM - len(g))
    return np.asarray(g[:G_NUM], np.float32)


def _left(x):
    if x == INF or x is None:
        return 0.0, 1.0
    return min(20, int(x)) / 10.0, 0.0


def _card(c):
    return {k: c[k] for k in ("id", "u", "c", "g") if k in c}


def encode(o, f):
    """The input arrays of observation O; F is a Featurizer."""
    v = f.v
    stance = (o.get("stance") or [None, 0])[0]
    g_cat = [v("plan", o["exam"].get("plan")), v("step", o["exam"].get("step")),
             v("produce", o["exam"].get("produce")), v("stance", stance)]
    st = o.get("status") or []
    st_kind = np.zeros((1, max(1, len(st))), np.int64)
    st_num = np.zeros((1, max(1, len(st)), 3), np.float32)
    for i, s in enumerate(st):
        turns = 20 if s[2] == INF and s[0] in _LONG_WHEN_UNLIMITED else s[2]
        left, inf = _left(turns)
        st_kind[0, i] = v("status", s[0])
        st_num[0, i] = (slog(s[1]), left, inf)
    things = []
    for i, c in enumerate(o["hand"]):
        things.append((KIND_HAND, f.card(_card(c)), (min(i, 9) / 9.0, float(c.get("p", 0)), 0, 0, 0, 0)))
    for d in o.get("drinks") or ():
        things.append((KIND_DRINK, f.drink(d), (0, 1.0, 0, 0, 0, 0)))
    for zone, kind in ZONES:
        for c in o.get(zone) or ():
            things.append((kind, f.card(_card(c)), (0, 0, 0, 0, 0, 0)))
    for e in o.get("ench") or ():
        ul, ui = _left(e[1])
        tl, ti = _left(e[2])
        things.append((KIND_ENCH, f.enchant(e[0]), (0, 0, ul, ui, tl, ti)))
    T = max(1, len(things))
    t_kind = np.zeros((1, T), np.int64)
    t_cat = np.zeros((1, T, len(THING_CAT)), np.int64)
    t_num = np.zeros((1, T, len(THING_NUM)), np.float32)
    s_cat = np.zeros((1, T, SLOTS, len(SLOT_CAT)), np.int64)
    s_num = np.zeros((1, T, SLOTS, len(SLOT_NUM)), np.float32)
    t_ext = np.zeros((1, T, T_EXT), np.float32)
    for i, (kind, (tc, tn, sc, sn), ext) in enumerate(things):
        t_kind[0, i], t_cat[0, i], t_num[0, i], s_cat[0, i], s_num[0, i] = kind, tc, tn, sc, sn
        t_ext[0, i] = ext
    n_hand = len(o["hand"])
    a = [int(x[1]) if x[0] == "play" else n_hand + int(x[1]) if x[0] == "drink" else -1 for x in o["legal"]]
    a_tok = np.full((1, max(1, len(a))), -2, np.int64)
    a_tok[0, :len(a)] = a
    return {"g_num": global_features(o)[None], "g_cat": np.asarray([g_cat], np.int64), "st_kind": st_kind,
            "st_num": st_num, "t_kind": t_kind, "t_cat": t_cat, "t_num": t_num, "s_cat": s_cat, "s_num": s_num,
            "t_ext": t_ext, "a_tok": a_tok}
